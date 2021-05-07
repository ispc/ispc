/*
  Copyright (c) 2010-2021, Intel Corporation
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
#include "target_registry.h"
#include "type.h"
#include "util.h"

#include <cstdarg>
#include <stdio.h>
#include <stdlib.h>
#ifdef ISPC_HOST_IS_WINDOWS
#include <time.h>
#else
#include <unistd.h>
#endif // ISPC_HOST_IS_WINDOWS

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ToolOutputFile.h>

using namespace ispc;

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

static void lPrintVersion() {
#ifdef ISPC_HOST_IS_WINDOWS
    printf("Intel(r) Implicit SPMD Program Compiler (Intel(r) ISPC), %s (build date %s, LLVM %s)\n"
           "Supported Visual Studio versions: %s.\n",
           ISPC_VERSION, BUILD_DATE, ISPC_LLVM_VERSION_STRING, ISPC_VS_VERSION);
#else
    printf("Intel(r) Implicit SPMD Program Compiler (Intel(r) ISPC), %s (build %s @ %s, LLVM %s)\n", ISPC_VERSION,
           BUILD_VERSION, BUILD_DATE, ISPC_LLVM_VERSION_STRING);
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

[[noreturn]] static void usage(int ret) {
    lPrintVersion();
    printf("\nusage: ispc\n");
    printf("    [--addressing={32,64}]\t\tSelect 32- or 64-bit addressing. (Note that 32-bit\n");
    printf("                          \t\taddressing calculations are done by default, even\n");
    printf("                          \t\ton 64-bit target architectures.)\n");
    printf("    [--arch={%s}]\t\tSelect target architecture\n", g->target_registry->getSupportedArchs().c_str());
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
    printf("    [--dwarf-version={2,3,4}]\t\tGenerate source-level debug information with given DWARF version "
           "(triggers -g).  Ignored for Windows target\n");
    printf("    [--emit-asm]\t\t\tGenerate assembly language file as output\n");
    printf("    [--emit-llvm]\t\t\tEmit LLVM bitcode file as output\n");
    printf("    [--emit-llvm-text]\t\t\tEmit LLVM bitcode file as output in textual form\n");
    printf("    [--emit-obj]\t\t\tGenerate object file file as output (default)\n");
#ifdef ISPC_GENX_ENABLED
    printf("    [--emit-spirv]\t\t\tGenerate SPIR-V file as output\n");
    // AOT compilation is temporary disabled on Windows
#ifndef ISPC_HOST_IS_WINDOWS
    printf("    [--emit-zebin]\t\t\tGenerate L0 binary as output\n");
#endif
#endif
    printf("    [--enable-llvm-intrinsics]\t\tEnable experimental feature to call LLVM intrinsics from ISPC "
           "source code\n");
    printf("    [--error-limit=<value>]\t\tLimit maximum number of errors emitting by ISPC to <value>\n");
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
    printf("        disable-zmm\t\tDisable using zmm registers for avx512 targets in favour of ymm. This also affects "
           "ABI.\n");
#ifdef ISPC_GENX_ENABLED
    printf("        emit-genx-hardware-mask\t\tEnable emitting of GenX implicit hardware mask.\n");
    printf("        enable-genx-foreach-varying\t\tEnable experimental foreach support inside varying control flow.\n");
#endif
    printf("        fast-masked-vload\t\tFaster masked vector loads on SSE (may go past end of array)\n");
    printf("        fast-math\t\t\tPerform non-IEEE-compliant optimizations of numeric expressions\n");
    printf("        force-aligned-memory\t\tAlways issue \"aligned\" vector load and store instructions\n");
    printf("    [--pic]\t\t\t\tGenerate position-independent code.  Ignored for Windows target\n");
    printf("    [--quiet]\t\t\t\tSuppress all output\n");
    printf("    [--support-matrix]\t\t\tPrint full matrix of supported targets, architectures and OSes\n");
    printf("    ");
    char targetHelp[2048];
    snprintf(targetHelp, sizeof(targetHelp),
             "[--target=<t>]\t\t\tSelect target ISA and width.\n"
             "<t>={%s}",
             g->target_registry->getSupportedTargets().c_str());
    PrintWithWordBreaks(targetHelp, 24, TerminalWidth(), stdout);
    printf("    ");
    snprintf(targetHelp, sizeof(targetHelp), "[--target-os=<os>]\t\t\tSelect target OS.  <os>={%s}",
             g->target_registry->getSupportedOSes().c_str());
    PrintWithWordBreaks(targetHelp, 24, TerminalWidth(), stdout);
    printf("    [--time-trace]\t\t\tTurn on time profiler. Generates JSON file based on output filename.\n");
    printf("    [--time-trace-granularity=<value>\tMinimum time granularity (in microseconds) traced by time "
           "profiler.\n");
    printf("    [--vectorcall/--no-vectorcall]\tEnable/disable vectorcall calling convention on Windows (x64 only). "
           "Disabled by default\n");
    printf("    [--version]\t\t\t\tPrint ispc version\n");
#ifdef ISPC_GENX_ENABLED
    printf("    [--vc-options=<\"-option1 -option2...\">]\t\t\t\tPass additional options to Vector Compiler backend\n");
#endif
    printf("    [--werror]\t\t\t\tTreat warnings as errors\n");
    printf("    [--woff]\t\t\t\tDisable warnings\n");
    printf("    [--wno-perf]\t\t\tDon't issue warnings related to performance-related issues\n");
    printf("    [--x86-asm-syntax=<option>]\t\tSelect style of code if generating assembly\n");
    printf("        intel\t\t\t\tEmit Intel-style assembly\n");
    printf("        att\t\t\t\tEmit AT&T-style assembly\n");
    printf("    [@<filename>]\t\t\tRead additional arguments from the given file\n");
    printf("    <file to compile or \"-\" for stdin>\n");
    exit(ret);
}

[[noreturn]] static void devUsage(int ret) {
    lPrintVersion();
    printf("\nusage (developer options): ispc\n");
    printf("    [--debug]\t\t\t\tPrint information useful for debugging ispc\n");
    printf("    [--debug-llvm]\t\t\tEnable LLVM debugging information (dumps to stderr)\n");
#ifndef ISPC_NO_DUMPS
    printf("    [--debug-phase=<value>]\t\tSet optimization phases to dump. "
           "--debug-phase=first,210:220,300,305,310:last\n");
#endif
    printf("    [--[no-]discard-value-names]\tDo not discard/Discard value names when generating LLVM IR.\n");
#ifndef ISPC_NO_DUMPS
    printf("    [--dump-file]\t\t\tDump module IR to file(s) in current directory\n");
#endif
    printf("    [--fuzz-seed=<value>]\t\tSeed value for RNG for fuzz testing\n");
    printf("    [--fuzz-test]\t\t\tRandomly perturb program input to test error conditions\n");
    printf("    [--off-phase=<value>]\t\tSwitch off optimization phases. --off-phase=first,210:220,300,305,310:last\n");
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
#ifdef ISPC_GENX_ENABLED
    printf("        disable-genx-gather-coalescing\t\tDisable GenX gather coalescing.\n");
#endif
    printf("    [--print-target]\t\t\tPrint target's information\n");
    printf("    [--yydebug]\t\t\t\tPrint debugging information during parsing\n");
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
        bool insideDQ = false;
        bool insideSQ = false;
        std::string arg;
        char c = GetNextChar();

        // First consume any white-space before the argument
        while (isspace(c))
            c = GetNextChar();

        if (c == '\0')
            // Reached the end so no more arguments
            return NULL;

        // c now has the first character of the next argument, so collect the rest
        while (c != '\0' && !(isspace(c) && !insideDQ && !insideSQ)) {
            if (c == '\"' && !insideSQ) {
                c = GetNextChar();
                insideDQ = !insideDQ;
                continue;
            }
            if (c == '\'' && !insideDQ) {
                c = GetNextChar();
                insideSQ = !insideSQ;
                continue;
            }
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
static void lAddSingleArg(char *arg, std::vector<char *> &argv);

/** Add all args from a given factory to the argv passed as parameters, which could
 *  include recursing into another ArgFactory.
 */
static void lAddArgsFromFactory(ArgFactory &Args, std::vector<char *> &argv) {
    while (true) {
        char *NextArg = Args.GetNextArg();
        if (NextArg == NULL)
            break;
        lAddSingleArg(NextArg, argv);
    }
}

/** Parse an open file for arguments and add them to the argv passed as parameters */
static void lAddArgsFromFile(FILE *file, std::vector<char *> &argv) {
    FileArgFactory args(file);
    lAddArgsFromFactory(args, argv);
}

/** Parse a string for arguments and add them to the argv passed as parameters */
static void lAddArgsFromString(const char *string, std::vector<char *> &argv) {
    StringArgFactory args(string);
    lAddArgsFromFactory(args, argv);
}

/** Add a single argument to the argv passed as parameters. If the argument is of the
 *  form @<filename> and <filename> exists and is readable, the arguments in the file will be
 *  inserted into argv in place of the original argument.
 */
static void lAddSingleArg(char *arg, std::vector<char *> &argv) {
    if (arg[0] == '@') {
        char *filename = &arg[1];
        FILE *file = fopen(filename, "r");
        if (file != NULL) {
            lAddArgsFromFile(file, argv);
            fclose(file);
            arg = NULL;
        }
    }
    if (arg != NULL) {
        argv.push_back(arg);
    }
}

/** We take arguments from both the command line as well as from the
 *  ISPC_ARGS environment variable - and each of these can include a file containing
 *  additional arguments using @<filename>. This function returns a new set of
 *  arguments representing the ones from all these sources merged together.
 */
static void lGetAllArgs(int Argc, char *Argv[], std::vector<char *> &argv) {
    // Copy over the command line arguments (passed in)
    for (int i = 0; i < Argc; ++i)
        lAddSingleArg(Argv[i], argv);

    // See if we have any set via the environment variable
    const char *env = getenv("ISPC_ARGS");
    if (env)
        lAddArgsFromString(env, argv);
}

static void lSignal(void *) { FATAL("Unhandled signal sent to process; terminating."); }

// ArgErrors accumulates error and warning messages during arguments parsing
// prints them after the parsing is done. We need to delay printing to take
// into account such options as --quite, --nowrap --werror, which affects how
// errors and warnings are treated and printed.
class ArgErrors {
    enum class MsgType { warning, error };
    std::vector<std::pair<MsgType, std::string>> m_messages;
    void AddMessage(MsgType msg_type, const char *format, va_list args) {
        char *messageBuf;
        if (vasprintf(&messageBuf, format, args) == -1) {
            fprintf(stderr, "vasprintf() unable to allocate memory!\n");
            exit(-1);
        }

        m_messages.push_back(std::make_pair(msg_type, messageBuf));

        free(messageBuf);
    }

  public:
    ArgErrors(){};
    void AddError(const char *format, ...) PRINTF_FUNC {
        va_list args;
        va_start(args, format);
        AddMessage(MsgType::error, format, args);
        va_end(args);
    }
    void AddWarning(const char *format, ...) PRINTF_FUNC {
        va_list args;
        va_start(args, format);
        AddMessage(MsgType::warning, format, args);
        va_end(args);
    }
    void Emit() {
        bool errors = false;
        for (auto &message : m_messages) {
            if (message.first == MsgType::error || g->warningsAsErrors) {
                errors = true;
                Error(SourcePos(), "%s", message.second.c_str());
            } else {
                Warning(SourcePos(), "%s", message.second.c_str());
            }
        }
        if (errors) {
            exit(-1);
        }
    }
};

static int ParsingPhaseName(char *stage, ArgErrors &errorHandler) {
    if (strncmp(stage, "first", 5) == 0) {
        return 0;
    } else if (strncmp(stage, "last", 4) == 0) {
        return LAST_OPT_NUMBER;
    } else {
        int t = atoi(stage);
        if (t < 0 || t > LAST_OPT_NUMBER) {
            errorHandler.AddError("Phases must be from 0 to %d. %s is incorrect.", LAST_OPT_NUMBER, stage);
            return 0;
        } else {
            return t;
        }
    }
}

enum class VectorCallStatus { none, enabled, disabled };

static void setCallingConv(VectorCallStatus vectorCall, Arch arch) {
    // Restrict vectorcall to just x86_64 - vectorcall for x86 not supported yet.
    if (g->target_os == TargetOS::windows && vectorCall == VectorCallStatus::enabled &&
        // Arch is not properly set yet, we assume none is x86_64.
        (arch == Arch::x86_64 || arch == Arch::none)) {
        g->calling_conv = CallingConv::x86_vectorcall;
    } else {
        g->calling_conv = CallingConv::defaultcall;
    }
}

static void writeCompileTimeFile(const char *outFileName) {
    llvm::SmallString<128> jsonFileName(outFileName);
    jsonFileName.append(".json");
    llvm::sys::fs::OpenFlags flags = llvm::sys::fs::OF_Text;
    std::error_code error;
    std::unique_ptr<llvm::ToolOutputFile> of(new llvm::ToolOutputFile(jsonFileName.c_str(), error, flags));

    if (error) {
        Error(SourcePos(), "Cannot open json file \"%s\".\n", jsonFileName.c_str());
    }

    llvm::raw_fd_ostream &fos(of->os());
    llvm::timeTraceProfilerWrite(fos);
    of->keep();
    return;
}

static std::set<int> ParsingPhases(char *stages, ArgErrors &errorHandler) {
    constexpr int parsing_limit = 100;
    std::set<int> phases;
    auto len = strnlen(stages, parsing_limit);
    if (len == 0) {
        errorHandler.AddError("Empty phase list.");
        return phases;
    }
    if (len == parsing_limit && stages[parsing_limit] != '\0') {
        errorHandler.AddError("Phase list is too long.");
        return phases;
    }
    int begin = ParsingPhaseName(stages, errorHandler);
    int end = begin;

    for (unsigned i = 0; i < strlen(stages); i++) {
        if ((stages[i] == ',') || (i == strlen(stages) - 1)) {
            for (int j = begin; j < end + 1; j++) {
                phases.insert(j);
            }
            begin = ParsingPhaseName(stages + i + 1, errorHandler);
            end = begin;
        } else if (stages[i] == ':') {
            end = ParsingPhaseName(stages + i + 1, errorHandler);
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

extern int yydebug;

int main(int Argc, char *Argv[]) {
    std::vector<char *> argv;
    lGetAllArgs(Argc, Argv, argv);
    int argc = argv.size();

    llvm::sys::AddSignalHandler(lSignal, NULL);

    // initialize available LLVM targets
#ifdef ISPC_X86_ENABLED
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();
    LLVMInitializeX86TargetMC();
#endif

#ifdef ISPC_ARM_ENABLED
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

#ifdef ISPC_WASM_ENABLED
    LLVMInitializeWebAssemblyAsmParser();
    LLVMInitializeWebAssemblyAsmPrinter();
    LLVMInitializeWebAssemblyDisassembler();
    LLVMInitializeWebAssemblyTarget();
    LLVMInitializeWebAssemblyTargetInfo();
    LLVMInitializeWebAssemblyTargetMC();
#endif
    char *file = NULL;
    const char *headerFileName = NULL;
    const char *outFileName = NULL;
    const char *depsFileName = NULL;
    const char *depsTargetName = NULL;
    const char *hostStubFileName = NULL;
    const char *devStubFileName = NULL;
    // Initiailize globals early so that we can set various option values
    // as we're parsing below
    g = new Globals;

    Module::OutputType ot = Module::Object;
    Module::OutputFlags flags = Module::NoFlags;
    Arch arch = Arch::none;
    std::vector<ISPCTarget> targets;
    const char *cpu = NULL, *intelAsmSyntax = NULL;
    VectorCallStatus vectorCall = VectorCallStatus::none;

    ArgErrors errorHandler;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help")) {
            usage(0);
        } else if (!strcmp(argv[i], "--help-dev")) {
            devUsage(0);
        } else if (!strcmp(argv[i], "--support-matrix")) {
            g->target_registry->printSupportMatrix();
            exit(0);
        } else if (!strncmp(argv[i], "-D", 2)) {
            g->cppArgs.push_back(argv[i]);
        } else if (!strncmp(argv[i], "--addressing=", 13)) {
            if (atoi(argv[i] + 13) == 64)
                // FIXME: this doesn't make sense on 32 bit platform.
                g->opt.force32BitAddressing = false;
            else if (atoi(argv[i] + 13) == 32)
                g->opt.force32BitAddressing = true;
            else {
                errorHandler.AddError("Addressing width \"%s\" invalid -- only 32 and "
                                      "64 are allowed.",
                                      argv[i] + 13);
            }
        } else if (!strncmp(argv[i], "--arch=", 7)) {
            Arch prev_arch = arch;

            arch = ParseArch(argv[i] + 7);
            if (arch == Arch::error) {
                errorHandler.AddError("Unsupported value for --arch, supported values are: %s",
                                      g->target_registry->getSupportedArchs().c_str());
            }

            if (prev_arch != Arch::none && prev_arch != arch) {
                std::string prev_arch_str = ArchToString(prev_arch);
                std::string arch_str = ArchToString(arch);
                errorHandler.AddWarning("Overwriting --arch=%s with --arch=%s", prev_arch_str.c_str(),
                                        arch_str.c_str());
            }
        } else if (!strncmp(argv[i], "--x86-asm-syntax=", 17)) {
            intelAsmSyntax = argv[i] + 17;
            if (!((std::string(intelAsmSyntax) == "intel") || (std::string(intelAsmSyntax) == "att"))) {
                intelAsmSyntax = NULL;
                errorHandler.AddError("Invalid value for --x86-asm-syntax: \"%s\" -- "
                                      "only intel and att are allowed.",
                                      argv[i] + 17);
            }
        } else if (!strncmp(argv[i], "--cpu=", 6)) {
            cpu = argv[i] + 6;
        } else if (!strcmp(argv[i], "--fast-math")) {
            errorHandler.AddError("--fast-math option has been renamed to --opt=fast-math!");
        } else if (!strcmp(argv[i], "--fast-masked-vload")) {
            errorHandler.AddError("--fast-masked-vload option has been renamed to "
                                  "--opt=fast-masked-vload!");
        } else if (!strcmp(argv[i], "--debug"))
            g->debugPrint = true;
        else if (!strcmp(argv[i], "--debug-llvm"))
            llvm::DebugFlag = true;
        else if (!strcmp(argv[i], "--discard-value-names"))
            g->ctx->setDiscardValueNames(true);
        else if (!strcmp(argv[i], "--no-discard-value-names"))
            g->ctx->setDiscardValueNames(false);
        else if (!strcmp(argv[i], "--dllexport"))
            g->dllExport = true;
        else if (!strncmp(argv[i], "--dwarf-version=", 16)) {
            int val = atoi(argv[i] + 16);
            if (2 <= val && val <= 4) {
                g->generateDebuggingSymbols = true;
                g->generateDWARFVersion = val;
            } else {
                errorHandler.AddError("Invalid value for DWARF version: \"%s\" -- "
                                      "only 2, 3 and 4 are allowed.",
                                      argv[i] + 16);
            }
        } else if (!strcmp(argv[i], "--print-target"))
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
        else if (!strcmp(argv[i], "--emit-llvm"))
            ot = Module::Bitcode;
        else if (!strcmp(argv[i], "--emit-llvm-text"))
            ot = Module::BitcodeText;
        else if (!strcmp(argv[i], "--emit-obj"))
            ot = Module::Object;
#ifdef ISPC_GENX_ENABLED
        else if (!strcmp(argv[i], "--emit-spirv")) {
            ot = Module::SPIRV;
            // AOT compilation is temporary disabled on Windows
#ifndef ISPC_HOST_IS_WINDOWS
        } else if (!strcmp(argv[i], "--emit-zebin")) {
            ot = Module::ZEBIN;
#endif
        }
#endif
        else if (!strcmp(argv[i], "--enable-llvm-intrinsics")) {
            g->enableLLVMIntrinsics = true;
        } else if (!strcmp(argv[i], "-I")) {
            if (++i != argc) {
                lParseInclude(argv[i]);
            } else {
                errorHandler.AddError("No path specified after -I option.");
            }
        } else if (!strncmp(argv[i], "-I", 2))
            lParseInclude(argv[i] + 2);
        else if (!strcmp(argv[i], "--fuzz-test"))
            g->enableFuzzTest = true;
        else if (!strncmp(argv[i], "--fuzz-seed=", 12))
            g->fuzzTestSeed = atoi(argv[i] + 12);
        else if (!strcmp(argv[i], "--target")) {
            // FIXME: should remove this way of specifying the target...
            if (++i != argc) {
                auto result = ParseISPCTargets(argv[i]);
                targets = result.first;
                if (!result.second.empty()) {
                    errorHandler.AddError("Incorrect targets: %s.  Choices are: %s.", result.second.c_str(),
                                          g->target_registry->getSupportedTargets().c_str());
                }
            } else {
                errorHandler.AddError("No target specified after --target option.");
            }
        } else if (!strncmp(argv[i], "--target=", 9)) {
            auto result = ParseISPCTargets(argv[i] + 9);
            targets = result.first;
            if (!result.second.empty()) {
                errorHandler.AddError("Incorrect targets: %s.  Choices are: %s.", result.second.c_str(),
                                      g->target_registry->getSupportedTargets().c_str());
            }
        } else if (!strncmp(argv[i], "--target-os=", 12)) {
            g->target_os = ParseOS(argv[i] + 12);
            if (g->target_os == TargetOS::error) {
                errorHandler.AddError("Unsupported value for --target-os, supported values are: %s",
                                      g->target_registry->getSupportedOSes().c_str());
            }
        } else if (!strcmp(argv[i], "--no-vectorcall")) {
            vectorCall = VectorCallStatus::disabled;
        } else if (!strcmp(argv[i], "--vectorcall")) {
            vectorCall = VectorCallStatus::enabled;
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
                errorHandler.AddError("Unknown --math-lib= option \"%s\".", lib);
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
            else if (!strcmp(opt, "disable-zmm"))
                g->opt.disableZMM = true;
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
#ifdef ISPC_GENX_ENABLED
            else if (!strcmp(opt, "disable-genx-gather-coalescing"))
                g->opt.disableGenXGatherCoalescing = true;
            else if (!strcmp(opt, "emit-genx-hardware-mask"))
                g->opt.emitGenXHardwareMask = true;
            else if (!strcmp(opt, "enable-genx-foreach-varying"))
                g->opt.enableForeachInsideVarying = true;
            else if (!strcmp(opt, "enable-genx-unsafe-masked-load"))
                g->opt.enableGenXUnsafeMaskedLoad = true;
#endif
            else {
                errorHandler.AddError("Unknown --opt= option \"%s\".", opt);
            }
        } else if (!strncmp(argv[i], "--force-alignment=", 18)) {
            g->forceAlignment = atoi(argv[i] + 18);
        } else if (!strcmp(argv[i], "--time-trace")) {
            g->enableTimeTrace = true;
        } else if (!strncmp(argv[i], "--time-trace-granularity=", 25)) {
            g->timeTraceGranularity = atoi(argv[i] + 25);
        } else if (!strcmp(argv[i], "--woff") || !strcmp(argv[i], "-woff")) {
            g->disableWarnings = true;
            g->emitPerfWarnings = false;
        } else if (!strcmp(argv[i], "--werror"))
            g->warningsAsErrors = true;
        else if (!strncmp(argv[i], "--error-limit=", 14)) {
            int errLimit = atoi(argv[i] + 14);
            if (errLimit >= 0)
                g->errorLimit = errLimit;
            else
                errorHandler.AddError("Invalid value for --error-limit: \"%d\" -- "
                                      "value cannot be a negative number.",
                                      errLimit);
        } else if (!strcmp(argv[i], "--nowrap"))
            g->disableLineWrap = true;
        else if (!strcmp(argv[i], "--wno-perf") || !strcmp(argv[i], "-wno-perf"))
            g->emitPerfWarnings = false;
        else if (!strcmp(argv[i], "-o")) {
            if (++i != argc) {
                outFileName = argv[i];
            } else {
                errorHandler.AddError("No output file specified after -o option.");
            }
        } else if (!strncmp(argv[i], "--outfile=", 10))
            outFileName = argv[i] + strlen("--outfile=");
        else if (!strcmp(argv[i], "-h")) {
            if (++i != argc) {
                headerFileName = argv[i];
            } else {
                errorHandler.AddError("No header file name specified after -h option.");
            }
        } else if (!strncmp(argv[i], "--header-outfile=", 17)) {
            headerFileName = argv[i] + strlen("--header-outfile=");
        } else if (!strcmp(argv[i], "-O0")) {
            g->opt.level = 0;
            g->codegenOptLevel = Globals::CodegenOptLevel::None;
        } else if (!strcmp(argv[i], "-O") || !strcmp(argv[i], "-O1") || !strcmp(argv[i], "-O2") ||
                   !strcmp(argv[i], "-O3")) {
            g->opt.level = 1;
            g->codegenOptLevel = Globals::CodegenOptLevel::Aggressive;
            if (!strcmp(argv[i], "-O1"))
                g->opt.disableCoherentControlFlow = true;
        } else if (!strcmp(argv[i], "-")) {
            file = argv[i];
        } else if (!strcmp(argv[i], "--nostdlib"))
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
            yydebug = 1;
        } else if (!strcmp(argv[i], "-MMM")) {
            if (++i != argc) {
                depsFileName = argv[i];
                flags |= Module::GenerateFlatDeps;
            } else {
                errorHandler.AddError("No output file name specified after -MMM option.");
            }
        } else if (!strcmp(argv[i], "-M")) {
            flags |= Module::GenerateMakeRuleForDeps | Module::OutputDepsToStdout;
        } else if (!strcmp(argv[i], "-MF")) {
            depsFileName = nullptr;
            if (++i != argc) {
                depsFileName = argv[i];
            } else {
                errorHandler.AddError("No output file name specified after -MF option.");
            }
        } else if (!strcmp(argv[i], "-MT")) {
            depsTargetName = nullptr;
            if (++i != argc) {
                depsTargetName = argv[i];
            } else {
                errorHandler.AddError("No target name specified after -MT option.");
            }
        } else if (!strcmp(argv[i], "--dev-stub")) {
            if (++i != argc) {
                devStubFileName = argv[i];
            } else {
                errorHandler.AddError("No output file name specified after --dev-stub option.");
            }
        } else if (!strcmp(argv[i], "--host-stub")) {
            if (++i != argc) {
                hostStubFileName = argv[i];
            } else {
                errorHandler.AddError("No output file name specified after --host-stub option.");
            }
        }
#ifndef ISPC_NO_DUMPS
        else if (strncmp(argv[i], "--debug-phase=", 14) == 0) {
            errorHandler.AddWarning("Adding debug phases may change the way PassManager"
                                    "handles the phases and it may possibly make some bugs go"
                                    "away or introduce the new ones.");
            g->debug_stages = ParsingPhases(argv[i] + strlen("--debug-phase="), errorHandler);
        } else if (strncmp(argv[i], "--dump-file", 11) == 0)
            g->dumpFile = true;
#endif

        else if (strncmp(argv[i], "--off-phase=", 12) == 0) {
            g->off_stages = ParsingPhases(argv[i] + strlen("--off-phase="), errorHandler);
#ifdef ISPC_GENX_ENABLED
        } else if (!strncmp(argv[i], "--vc-options=", 12)) {
            g->vcOpts = argv[i] + strlen("--vc-options=");
#endif
        } else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--version")) {
            lPrintVersion();
            return 0;
        } else if (argv[i][0] == '-') {
            errorHandler.AddError("Unknown option \"%s\".", argv[i]);
        } else {
            if (file != NULL) {
                errorHandler.AddError("Multiple input files specified on command "
                                      "line: \"%s\" and \"%s\".",
                                      file, argv[i]);
            } else {
                file = argv[i];
            }
        }
    }

    // Emit accumulted errors and warnings, if any.
    // All the rest of errors and warnigns will be processed in regullar way.
    errorHandler.Emit();

    if (file == NULL) {
        Error(SourcePos(), "No input file were specified. To read text from stdin use \"-\" as file name.");
        exit(1);
    }

    // Default settings for PS4
    if (g->target_os == TargetOS::ps4) {
        flags |= Module::GeneratePIC;
        if (!cpu) {
            // Default is btver2, but do not enforce it.
            cpu = "btver2";
        }
        /*
        if (cpu && std::string(cpu) != "btver2" && std::string(cpu) != "ps4") {
            Warning(SourcePos(), "--cpu switch is ignored for PS4 target OS. btver2 (ps4) cpu is used.");
        }
        */
        if (arch != Arch::x86_64) {
            Warning(SourcePos(), "--arch switch is ignored for PS4 target OS. x86-64 arch is used.");
            arch = Arch::x86_64;
        }
    }

    // Default setting for "custom_linux"
    if (g->target_os == TargetOS::custom_linux) {
        flags |= Module::GeneratePIC;
        if (!cpu) {
            cpu = "cortex-a57";
        }
        if (targets.empty()) {
            targets.push_back(ISPCTarget::neon_i32x4);
            std::string target_string = ISPCTargetToString(targets[0]);
            Warning(SourcePos(),
                    "No --target specified on command-line."
                    " Using \"%s\".",
                    target_string.c_str());
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
        devStubFileName == NULL) {
        Warning(SourcePos(), "No output file or header file name specified. "
                             "Program will be compiled and warnings/errors will "
                             "be issued, but no output will be generated.");
    }

    if (g->target_os == TargetOS::windows && (flags & Module::GeneratePIC) != 0) {
        Warning(SourcePos(), "--pic switch for Windows target will be ignored.");
    }

    if (g->target_os != TargetOS::windows && g->dllExport) {
        Warning(SourcePos(), "--dllexport switch will be ignored, as the target OS is not Windows.");
    }

    if (vectorCall != VectorCallStatus::none &&
        (g->target_os != TargetOS::windows ||
         // This is a hacky check. Arch is properly set later, so we rely that default means x86_64.
         (arch != Arch::x86_64 && arch != Arch::none))) {
        Warning(SourcePos(), "--vectorcall/--no-vectorcall are supported only for x86_64 Windows target, so these "
                             "options will be ignored.");
    }

    if (targets.size() > 1)
        g->isMultiTargetCompilation = true;

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

    for (auto target : targets) {
        if (target == ISPCTarget::wasm_i32x4) {
            Assert(targets.size() == 1 && "wasm32 supports only one target: i32x4");
            arch = Arch::wasm32;
            g->target_os = TargetOS::web;
        }
#ifdef ISPC_GENX_ENABLED
        if (ISPCTargetIsGen(target)) {
            Assert(targets.size() == 1 && "multi-target is not supported for genx-* targets yet.");
            // Generate .spv for gen target instead of object by default.
            if (ot == Module::Object) {
                Warning(SourcePos(), "Emitting spir-v file for genx-* targets.");
                ot = Module::SPIRV;
            }
        }
#endif
    }

    // This needs to happen after the TargetOS is decided.
    setCallingConv(vectorCall, arch);
    if (g->enableTimeTrace) {
        llvm::timeTraceProfilerInitialize(g->timeTraceGranularity, "ispc");
    }
    int ret = 0;
    {
        llvm::TimeTraceScope TimeScope("ExecuteCompiler");
        ret = Module::CompileAndOutput(file, arch, cpu, targets, flags, ot, outFileName, headerFileName, depsFileName,
                                       depsTargetName, hostStubFileName, devStubFileName);
    }

    if (g->enableTimeTrace) {
        // Write to file only if compilation is successfull.
        if ((ret == 0) && (outFileName != NULL)) {
            writeCompileTimeFile(outFileName);
        }
        llvm::timeTraceProfilerCleanup();
    }
    return ret;
}
