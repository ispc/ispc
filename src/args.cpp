/*
  Copyright (c) 2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file args.cpp
    @brief Command line argument parsing implementation for ispc
*/

#include "args.h"
#include "ispc.h"
#include "ispc_version.h"
#include "module.h"
#include "target_registry.h"
#include "util.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>

#ifdef ISPC_HOST_IS_WINDOWS
#define strcasecmp stricmp
#ifndef BUILD_DATE
#define BUILD_DATE __DATE__
#endif
#if _MSC_VER >= 1900
#define ISPC_VS_VERSION "Visual Studio 2015 and later"
#else
#define ISPC_VS_VERSION "Visual Studio 2013 and earlier"
#endif
#endif // ISPC_HOST_IS_WINDOWS

using namespace ispc;

// For boolean command line options there are actually three states. Beyond "on" and "off", there's "unspecified",
// which might trigger a different default behavior.
enum class BooleanOptValue { none, enabled, disabled };

static void lPrintVersion() {
    printf("%s\n", ISPC_VERSION_STRING);
#ifdef ISPC_HOST_IS_WINDOWS
    printf("Supported Visual Studio versions: %s.\n", ISPC_VS_VERSION);
#endif
}

static ArgsParseResult usage() {
    lPrintVersion();
    printf("\nusage: ispc\n");
    printf("    [--addressing={32,64}]\t\tSelect 32- or 64-bit addressing. (Note that 32-bit addressing calculations "
           "are done by default, even on 64-bit target architectures.)\n");
    printf("    [--arch={%s}]\t\tSelect target architecture\n", g->target_registry->getSupportedArchs().c_str());
#ifndef ISPC_HOST_IS_WINDOWS
    printf("    [--colored-output]\t\t\tAlways use terminal colors in error/warning messages\n");
#endif
    printf("    [--cpu=<type>]\t\t\tAn alias for [--device=<type>] switch\n");
    printf("    [-D<foo>]\t\t\t\t#define given value when running preprocessor\n");
#if defined(ISPC_MACOS_TARGET_ON) || defined(ISPC_IOS_TARGET_ON)
    printf("    [--darwin-version-min=<major.minor>]Set the minimum macOS/iOS version required for the "
           "deployment.\n");
#endif
    printf("    [-dD]\t\t\t\tPrint macro definitions in addition to the preprocessor result\n");
    printf("    [--dev-stub <filename>]\t\tEmit device-side offload stub functions to file\n");
    printf("    ");
    char cpuHelp[2048];
    snprintf(cpuHelp, sizeof(cpuHelp), "[--device=<type>]\t\t\tSelect target device\n<type>={%s}\n",
             Target::SupportedCPUs().c_str());
    PrintWithWordBreaks(cpuHelp, 16, TerminalWidth(), stdout);
    printf("    [--dllexport]\t\t\tMake non-static functions DLL exported.  Windows target only\n");
    printf("    [-dM]\t\t\t\tPrint macro definitions for the preprocessor result\n");
    printf("    [--dwarf-version={2,3,4,5}]\t\tGenerate source-level debug information with given DWARF version "
           "(triggers -g).  It forces the usage of DWARF debug info on Windows target\n");
    printf("    [-E]\t\t\t\tRun only the preprocessor\n");
    printf("    [--emit-asm]\t\t\tGenerate assembly language file as output\n");
    printf("    [--emit-llvm]\t\t\tEmit LLVM bitcode file as output\n");
    printf("    [--emit-llvm-text]\t\t\tEmit LLVM bitcode file as output in textual form\n");
    printf("    [--emit-obj]\t\t\tGenerate object file as output (default)\n");
#ifdef ISPC_XE_ENABLED
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
    printf("    [--sample-profiling-debug-info]\tGenerate debug info optimized for sample-based profiling\n");
    printf("    [--profile-sample-use=<file>]\t\tUse sample profile data for optimization\n");
    printf("    [--help]\t\t\t\tPrint help\n");
    printf("    [--help-dev]\t\t\tPrint help for developer options\n");
    printf("    [--host-stub <filename>]\t\tEmit host-side offload stub functions to file\n");
    printf("    [-h <name>/--header-outfile=<name>]\tOutput filename for header\n");
    printf("    [-I <path>]\t\t\t\tAdd <path> to #include file search path\n");
    printf(
        "    [--include-float16-conversions]\tAdd float16 conversion functions permanently to the compiled module\n");
    printf("    [--ignore-preprocessor-errors]\tSuppress errors from the preprocessor\n");
    printf("    [--instrument]\t\t\tEmit instrumentation to gather performance data\n");
    printf("    [--math-lib=<option>]\t\tSelect math library\n");
    printf("        default\t\t\t\tUse ispc's built-in math functions\n");
    printf("        fast\t\t\t\tUse high-performance but lower-accuracy math functions\n");
    printf("        svml\t\t\t\tUse the Intel(r) SVML math libraries\n");
    printf("        system\t\t\t\tUse the system's math library (*may be quite slow*)\n");
    printf("    [-f[no-]function-sections]\t\tPlace each function in its own section\n");
    printf("    [--mcmodel=<value>]\t\t\tDefine the code model to use for code generation\n");
    printf("        small\t\t\t\tThe program and its symbols must be linked in the lower 2GB of the address space "
           "(default)\n");
    printf("        large\t\t\t\tThe program has no assumption about addresses and sizes of sections\n");
    printf("    [-MMM <filename>]\t\t\tWrite #include dependencies to given file\n");
    printf("    [-M]\t\t\t\tOutput a rule suitable for `make' describing the dependencies of the main source file to "
           "stdout\n");
    printf("    [-MF <filename>]\t\t\tWhen used with `-M', specifies a file to write the dependencies to\n");
    printf("    [-MT <filename>]\t\t\tWhen used with `-M', changes the target of the rule emitted by dependency "
           "generation\n");
    printf("    [--nanobind-wrapper=<filename>]\tWrite a nanobind wrapper to given file\n");
    printf("    [--no-omit-frame-pointer]\t\tDisable frame pointer omission. It may be useful for profiling\n");
    printf("    [--nostdlib]\t\t\tDon't make the ispc standard library available\n");
    printf("    [--no-pragma-once]\t\t\tDon't use #pragma once in created headers\n");
    printf("    [-o <name>/--outfile=<name>]\tOutput filename (may be \"-\" for standard output)\n");
    printf("    [-O0/-O(1/2/3)]\t\t\tSet optimization level. Default behavior is to optimize for speed\n");
    printf("        -O0\t\t\t\tOptimizations disabled\n");
    printf("        -O1\t\t\t\tOptimization for size\n");
    printf("        -O2/O3\t\t\t\tOptimization for speed\n");
    printf("    [--opt=<option>]\t\t\tSet optimization option\n");
    printf("        disable-assertions\t\tRemove assertion statements from final code\n");
    printf("        disable-fma\t\t\tDisable 'fused multiply-add' instructions (on targets that support them)\n");
    printf("        disable-gathers\t\t\tDisable gathers generation on targets that support them\n");
    printf("        disable-scatters\t\tDisable scatters generation on targets that support them\n");
    printf("        disable-loop-unroll\t\tDisable loop unrolling\n");
    printf("        disable-zmm\t\t\tDisable using zmm registers for avx512skx-x16, avx512icl-x16 targets in favor of "
           "ymm. This also affects ABI\n");
#ifdef ISPC_XE_ENABLED
    printf("        emit-xe-hardware-mask\t\tEnable emitting of Xe implicit hardware mask\n");
    printf("        enable-xe-foreach-varying\t\tEnable experimental foreach support inside varying control flow\n");
#endif
    printf("        fast-masked-vload\t\tFaster masked vector loads on SSE (may go past end of array)\n");
    printf("        fast-math\t\t\tPerform non-IEEE-compliant optimizations of numeric expressions\n");
    printf("        force-aligned-memory\t\tAlways issue \"aligned\" vector load and store instructions\n");
    printf("        reset-ftz-daz\t\t\tReset FTZ/DAZ flags on ISPC extern function entrance / restore on return\n");
    printf("    [--pic]\t\t\t\tGenerate position-independent code.  Ignored for Windows target\n");
    printf("    [--PIC]\t\t\t\tGenerate position-independent code avoiding any limit on the size of the global offset "
           "table. Ignored for Windows target\n");
    printf("    [--quiet]\t\t\t\tSuppress all output\n");
    printf("    [--stack-protector]\t\t\tEnable stack protectors for functions with larger stack variables.\n");
    printf("    [--stack-protector=<option>]\tEnable stack protectors\n");
    printf("        all\t\t\t\tfor all functions.\n");
    printf("        none\t\t\t\tfor no functions (default).\n");
    printf("        on\t\t\t\tfor functions with larger stack variables (same as --stack-protector).\n");
    printf("        strong\t\t\t\tfor functions with stack variables of any size or taking addresses of local "
           "variables.\n");
    printf("    [--support-matrix]\t\t\tPrint full matrix of supported targets, architectures and OSes\n");
    printf("    ");
    char targetHelp[2048];
    snprintf(targetHelp, sizeof(targetHelp),
             "[--target=<t>]\t\t\tSelect target ISA and width\n"
             "<t>={%s}",
             g->target_registry->getSupportedTargets().c_str());
    PrintWithWordBreaks(targetHelp, 24, TerminalWidth(), stdout);
    printf("    ");
    snprintf(targetHelp, sizeof(targetHelp), "[--target-os=<os>]\t\t\tSelect target OS.  <os>={%s}",
             g->target_registry->getSupportedOSes().c_str());
    PrintWithWordBreaks(targetHelp, 24, TerminalWidth(), stdout);
    printf("    [--vectorcall/--no-vectorcall]\tEnable/disable vectorcall calling convention on Windows (x64 only). "
           "Disabled by default\n");
    printf("    [--version]\t\t\t\tPrint ispc version\n");
#ifdef ISPC_XE_ENABLED
    printf("    [--vc-options=<\"-option1 -option2...\">]\t\t\t\tPass additional options to Vector Compiler backend\n");
#endif
    printf("    [--werror]\t\t\t\tTreat warnings as errors\n");
    printf("    [--woff]\t\t\t\tDisable warnings\n");
    printf("    [--wno-perf]\t\t\tDon't issue warnings related to performance-related issues\n");
    printf("    [--[no-]wrap-signed-int]\t\t[Do not] preserve wraparound on signed integer overflow (default: do not "
           "preserve)\n");
    printf("    [--x86-asm-syntax=<option>]\t\tSelect style of code if generating assembly\n");
    printf("        intel\t\t\t\tEmit Intel-style assembly\n");
    printf("        att\t\t\t\tEmit AT&T-style assembly\n");
#ifdef ISPC_XE_ENABLED
    printf("    [--xe-stack-mem-size=<value>\t\tSet size of stateless stack memory in VC backend\n");
#endif
    printf("    [@<filename>]\t\t\tRead additional arguments from the given file\n");
    printf("    <file to compile or \"-\" for stdin>\n");
    return ArgsParseResult::help_requested;
}

static ArgsParseResult linkUsage() {
    lPrintVersion();
    printf("\nusage: ispc link\n");
    printf("\nLink several IR or SPIR-V files to selected output format: LLVM BC (default), LLVM text or SPIR-V\n");
    printf("    [--emit-llvm]\t\t\tEmit LLVM bitcode file as output\n");
    printf("    [--emit-llvm-text]\t\t\tEmit LLVM bitcode file as output in textual form\n");
#ifdef ISPC_XE_ENABLED
    printf("    [--emit-spirv]\t\t\tEmit SPIR-V file as output\n");
#endif
    printf("    [-o <name>/--outfile=<name>]\tOutput filename (may be \"-\" for standard output)\n");
    printf("    <files to link or \"-\" for stdin>\n");
    printf("\nExamples:\n");
    printf("    Link two SPIR-V files to LLVM BC output:\n");
    printf("        ispc link test_a.spv test_b.spv --emit-llvm -o test.bc\n");
    printf("    Link LLVM bitcode files to SPIR-V output:\n");
    printf("        ispc link test_a.bc test_b.bc --emit-spirv -o test.spv\n");
    return ArgsParseResult::help_requested;
}

static ArgsParseResult devUsage() {
    lPrintVersion();
    printf("\nusage (developer options): ispc\n");
    printf("    [--ast-dump]\t\tDump AST.\n");
    printf("    [--binary-type]\t\t\tPrint binary type (slim or composite).\n");
    printf("    [--debug]\t\t\t\tPrint information useful for debugging ispc\n");
    printf("    [--debug-llvm]\t\t\tEnable LLVM debugging information (dumps to stderr)\n");
    printf("    [--debug-pm]\t\t\tPrint verbose information from ispc pass manager\n");
    printf("    [--debug-pm-time-trace]\t\tPrint time tracing information from ispc pass manager\n");
    printf("    [--debug-phase=<value>]\t\tSet optimization or construction phases to dump. "
           "--debug-phase=pre:first,210:220,300,305,310:last\n");
    printf("    [--[no-]discard-value-names]\tDo not discard/Discard value names when generating LLVM IR\n");
    printf("    [--dump-file[=<path>]]\t\tDump module IR to file(s) in "
           "current directory, or to <path> if specified\n");
    printf("    [--gen-stdlib]\t\tEnable special compilation mode to generate LLVM IR for stdlib.ispc.\n");
    printf("    [--check-bitcode-libs]\t\tCheck the presence of bitcode libraries for ISPC slim binary.\n");
    printf("    [--nocpp]\t\t\t\tDon't run the C preprocessor\n");
    printf("    [--off-phase=<value>]\t\tSwitch off optimization phases. "
           "--off-phase=pre:first,210:220,300,305,310:last\n");
    printf("    [--opt=<option>]\t\t\tSet optimization option\n");
    printf("        enable-ldst-vectorizer\t\t\tEnable load/store vectorizer\n");
    printf("        enable-slp-vectorizer\t\t\tEnable SLP vectorizer\n");
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
#ifdef ISPC_XE_ENABLED
    printf("        disable-xe-gather-coalescing\t\tDisable Xe gather coalescing\n");
    printf("        threshold-for-xe-gather-coalescing=<0>\tMinimal number of eliminated memory instructions for "
           "Xe gather coalescing\n");
    printf("        build-llvm-loads-on-xe-gather-coalescing\t\tExperimental: build standard llvm loads on "
           "Xe gather coalescing\n");
    printf("        enable-xe-unsafe-masked-load\t\tEnable Xe unsafe masked load\n");
#endif
    printf("    [--time-trace]\t\t\tTurn on time profiler. Generates JSON file based on output filename\n");
    printf("    [--time-trace-granularity=<value>]\tMinimum time granularity (in microseconds) traced by time "
           "profiler\n");
    printf("    [--time-trace-pm]\t\t\tPrint time tracing information from ispc pass manager\n");
    printf("    [--print-target]\t\t\tPrint target's information\n");
    printf("    [--yydebug]\t\t\t\tPrint debugging information during parsing\n");
    return ArgsParseResult::help_requested;
}

/** Define an abstract base-class that implements the parsing of an character source and
 *  the breaking of it into the individual arguments
 */
class ArgFactory {
  private:
    /** Method provided by the derived classes to retrieve the next character from the stream.
     */
    virtual char GetNextChar() = 0;

  public:
    ArgFactory() {}
    virtual ~ArgFactory() {}

    char *GetNextArg() {
        bool insideDQ = false;
        bool insideSQ = false;
        std::string arg;
        char c = GetNextChar();

        // First consume any white-space before the argument
        while (isspace(c)) {
            c = GetNextChar();
        }

        if (c == '\0') {
            // Reached the end so no more arguments
            return nullptr;
        }

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

        return strdup(arg.c_str());
    }
};

/** Define a class to break the contents of an open file into the individual arguments */
class FileArgFactory : public ArgFactory {
  private:
    FILE *InputFile;

    char GetNextChar() override {
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

    char GetNextChar() override {
        char c = *InputString;

        if (c != '\0') {
            ++InputString;
        }

        return c;
    }

  public:
    StringArgFactory(const char *string) : InputString(string) {}
};

// Forward reference
static void lAddSingleArg(char *arg, std::vector<char *> &argv, bool duplicate);

/** Add all args from a given factory to the argv passed as parameters, which could
 *  include recursing into another ArgFactory.
 */
static void lAddArgsFromFactory(ArgFactory &Args, std::vector<char *> &argv) {
    while (true) {
        char *NextArg = Args.GetNextArg();
        if (NextArg == nullptr) {
            break;
        }
        lAddSingleArg(NextArg, argv, false);
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
static void lAddSingleArg(char *arg, std::vector<char *> &argv, bool duplicate) {
    if (arg[0] == '@') {
        char *filename = &arg[1];
        FILE *file = fopen(filename, "r");
        if (file != nullptr) {
            lAddArgsFromFile(file, argv);
            fclose(file);
            arg = nullptr;
        }
    }
    if (arg != nullptr) {
        if (duplicate) {
            // duplicate arg from main argv to make deallocation straightforward.
            argv.push_back(strdup(arg));
        } else {
            argv.push_back(arg);
        }
    }
}

/** We take arguments from both the command line as well as from the
 *  ISPC_ARGS environment variable - and each of these can include a file containing
 *  additional arguments using @<filename>. This function returns a new set of
 *  arguments representing the ones from all these sources merged together.
 */
void ispc::GetAllArgs(int Argc, char *Argv[], std::vector<char *> &argv) {
    // Copy over the command line arguments (passed in)
    for (int i = 0; i < Argc; ++i) {
        lAddSingleArg(Argv[i], argv, true);
    }

    // See if we have any set via the environment variable
    const char *env = getenv("ISPC_ARGS");
    if (env) {
        lAddArgsFromString(env, argv);
    }
}

void ispc::FreeArgv(std::vector<char *> &argv) {
    // argv vector consists of pointers to arguments as C strings alloced on
    // heap and collected form three source:  environment variable ISPC_ARGS,
    // @filename, inputs argv. They are needed to be deallocated.
    for (auto p : argv) {
        free(p);
    }
}

// ArgErrors accumulates error and warning messages during arguments parsing
// prints them after the parsing is done. We need to delay printing to take
// into account such options as --quite, --nowrap --werror, which affects how
// errors and warnings are treated and printed.
class ArgErrors {
    bool m_memoryError = false;
    enum class MsgType { warning, error };
    std::vector<std::pair<MsgType, std::string>> m_messages;
    void AddMessage(MsgType msg_type, const char *format, va_list args) {
        if (m_memoryError) {
            return; // Already in fallback mode
        }
        char *messageBuf = nullptr;
        if (vasprintf(&messageBuf, format, args) == -1) {
            fprintf(stderr, "vasprintf() unable to allocate memory!\n");
            m_memoryError = true;
            return;
        }

        m_messages.push_back(std::make_pair(msg_type, messageBuf));

        free(messageBuf);
    }

  public:
    ArgErrors() {};
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
    ArgsParseResult Emit() {
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
            return ArgsParseResult::failure;
        }
        return ArgsParseResult::success;
    }
};

static void setCallingConv(BooleanOptValue vectorCall, Arch arch) {
    // Restrict vectorcall to just x86_64 - vectorcall for x86 not supported yet.
    if (g->target_os == TargetOS::windows && vectorCall == BooleanOptValue::enabled &&
        // Arch is not properly set yet, we assume none is x86_64.
        (arch == Arch::x86_64 || arch == Arch::none)) {
        g->calling_conv = CallingConv::x86_vectorcall;
    } else {
        g->calling_conv = CallingConv::defaultcall;
    }
}

static std::string lParsePath(char *path, ArgErrors &errorHandler) {
    constexpr int parsing_limit = 1024;
    auto len = strnlen(path, parsing_limit);
    return std::string{path, len};
}

static int lParsingPhaseName(char *stage, ArgErrors &errorHandler) {
    if (strncmp(stage, "pre", 3) == 0) {
        return PRE_OPT_NUMBER;
    } else if (strncmp(stage, "first", 5) == 0) {
        return INIT_OPT_NUMBER;
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

static std::set<int> lParsingPhases(char *stages, ArgErrors &errorHandler) {
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
    int begin = lParsingPhaseName(stages, errorHandler);
    int end = begin;

    for (unsigned i = 0; i < strlen(stages); i++) {
        if ((stages[i] == ',') || (i == strlen(stages) - 1)) {
            for (int j = begin; j < end + 1; j++) {
                phases.insert(j);
            }
            begin = lParsingPhaseName(stages + i + 1, errorHandler);
            end = begin;
        } else if (stages[i] == ':') {
            end = lParsingPhaseName(stages + i + 1, errorHandler);
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
    std::stringstream ss(path);
    for (std::string item; std::getline(ss, item, delim);) {
        g->includePath.push_back(item);
    }
}

// Forward declarations for functions used in ParseCommandLineArgs
extern void printBinaryType();

ArgsParseResult ispc::ParseCommandLineArgs(int argc, char *argv[], std::string &file, Arch &arch, std::string &cpu,
                                           std::vector<ISPCTarget> &targets, Module::Output &output,
                                           std::vector<std::string> &linkFileNames, bool &isLinkMode) {
    BooleanOptValue vectorCall = BooleanOptValue::none;
    BooleanOptValue discardValueNames = BooleanOptValue::none;
    BooleanOptValue wrapSignedInt = BooleanOptValue::none;

    const char *intelAsmSyntax = nullptr;
    ArgErrors errorHandler;

    // If the first argument is "link"
    // ISPC will be used in a linkage mode
    if (argc > 1 && !strncmp(argv[1], "link", 4)) {
        isLinkMode = true;
        // Use bitcode format by default
        output.type = Module::Bitcode;

        if (argc < 2) {
            // Not sufficient number of arguments
            linkUsage();
            return ArgsParseResult::failure;
        }
        for (int i = 2; i < argc; ++i) {
            if (!strcmp(argv[i], "--help")) {
                return linkUsage();
            } else if (!strcmp(argv[i], "-o")) {
                if (++i != argc) {
                    output.out = argv[i];
                } else {
                    errorHandler.AddError("No output file specified after -o option.");
                }
            } else if (!strncmp(argv[i], "--outfile=", 10)) {
                output.out = argv[i] + strlen("--outfile=");
#ifdef ISPC_XE_ENABLED
            } else if (!strcmp(argv[i], "--emit-spirv")) {
                output.type = Module::SPIRV;
#endif
            } else if (!strcmp(argv[i], "--emit-llvm")) {
                output.type = Module::Bitcode;
            } else if (!strcmp(argv[i], "--emit-llvm-text")) {
                output.type = Module::BitcodeText;
            } else if (argv[i][0] == '-') {
                errorHandler.AddError("Unknown option \"%s\".", argv[i]);
            } else {
                file = argv[i];
                linkFileNames.push_back(file);
            }
        }
        // Emit accumulted errors and warnings, if any.
        if (auto result = errorHandler.Emit(); result != ArgsParseResult::success) {
            return result;
        }

        if (linkFileNames.size() == 0) {
            Error(SourcePos(), "No input files were specified.");
            return ArgsParseResult::failure;
        }

        if (output.out.empty()) {
            Warning(SourcePos(), "No output file name specified. "
                                 "The inputs will be linked and warnings/errors will "
                                 "be issued, but no output will be generated.");
        }

        return ArgsParseResult::success;
    }

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help")) {
            return usage();
        } else if (!strcmp(argv[i], "--help-dev")) {
            return devUsage();
        } else if (!strncmp(argv[i], "link", 4)) {
            errorHandler.AddError(
                "Option \"link\" can't be used in compilation mode. Use \"ispc link --help\" for details");
        } else if (!strcmp(argv[i], "--support-matrix")) {
            g->target_registry->printSupportMatrix();
            return ArgsParseResult::help_requested;
        } else if (!strncmp(argv[i], "-D", 2)) {
            g->cppArgs.push_back(argv[i]);
        } else if (!strncmp(argv[i], "--addressing=", 13)) {
            if (atoi(argv[i] + 13) == 64) {
                // FIXME: this doesn't make sense on 32 bit platform.
                g->opt.force32BitAddressing = false;
            } else if (atoi(argv[i] + 13) == 32) {
                g->opt.force32BitAddressing = true;
            } else {
                errorHandler.AddError("Addressing width \"%s\" invalid -- only 32 and "
                                      "64 are allowed.",
                                      argv[i] + 13);
            }
        } else if (!strncmp(argv[i], "--arch=", 7)) {
            Arch prev_arch = arch;

            arch = ParseArch(argv[i] + 7);
            if (arch == Arch::error) {
                errorHandler.AddError("Unsupported value for --arch, supported values are: %s. Use --support-matrix "
                                      "for complete information about supported targets, archs and target OSes.",
                                      g->target_registry->getSupportedArchs().c_str());
            }

            if (prev_arch != Arch::none && prev_arch != arch) {
                std::string prev_arch_str = ArchToString(prev_arch);
                std::string arch_str = ArchToString(arch);
                errorHandler.AddWarning("Overwriting --arch=%s with --arch=%s", prev_arch_str.c_str(),
                                        arch_str.c_str());
            }
        } else if (!strcmp(argv[i], "--ast-dump") || !strncmp(argv[i], "--ast-dump=", 11)) {
            g->astDump = Globals::ASTDumpKind::All;
        } else if (!strcmp(argv[i], "--binary-type")) {
            printBinaryType();
            return ArgsParseResult::help_requested;
        } else if (!strncmp(argv[i], "--x86-asm-syntax=", 17)) {
            intelAsmSyntax = argv[i] + 17;
            if (!((std::string(intelAsmSyntax) == "intel") || (std::string(intelAsmSyntax) == "att"))) {
                intelAsmSyntax = nullptr;
                errorHandler.AddError("Invalid value for --x86-asm-syntax: \"%s\" -- "
                                      "only intel and att are allowed.",
                                      argv[i] + 17);
            }
        } else if (!strncmp(argv[i], "--device=", 9)) {
            cpu = argv[i] + 9;
        } else if (!strncmp(argv[i], "--cpu=", 6)) {
            cpu = argv[i] + 6;
        } else if (!strcmp(argv[i], "--fast-math")) {
            errorHandler.AddError("--fast-math option has been renamed to --opt=fast-math!");
        } else if (!strcmp(argv[i], "--fast-masked-vload")) {
            errorHandler.AddError("--fast-masked-vload option has been renamed to "
                                  "--opt=fast-masked-vload!");
        } else if (!strcmp(argv[i], "--debug")) {
            g->debugPrint = true;
        } else if (!strcmp(argv[i], "--debug-llvm")) {
            llvm::DebugFlag = true;
        } else if (!strcmp(argv[i], "--debug-pm")) {
            g->debugPM = true;
        } else if (!strcmp(argv[i], "--debug-pm-time-trace") || !strcmp(argv[i], "--time-trace-pm")) {
            g->debugPMTimeTrace = true;
        } else if (!strcmp(argv[i], "--discard-value-names")) {
            discardValueNames = BooleanOptValue::enabled;
        } else if (!strcmp(argv[i], "--no-discard-value-names")) {
            discardValueNames = BooleanOptValue::disabled;
        } else if (!strcmp(argv[i], "--dllexport")) {
            g->dllExport = true;
        } else if (!strncmp(argv[i], "--dwarf-version=", 16)) {
            int val = atoi(argv[i] + 16);
            if (2 <= val && val <= 5) {
                g->generateDebuggingSymbols = true;
                g->generateDWARFVersion = val;
                g->debugInfoType = Globals::DebugInfoType::DWARF;
            } else {
                errorHandler.AddError("Invalid value for DWARF version: \"%s\" -- "
                                      "only 2, 3, 4 and 5 are allowed.",
                                      argv[i] + 16);
            }
#if defined(ISPC_MACOS_TARGET_ON) || defined(ISPC_IOS_TARGET_ON)
        } else if (!strncmp(argv[i], "--darwin-version-min=", 21)) {
            const char *version = argv[i] + 21;
            // Validate the version format
            std::string versionStr(version);
            llvm::VersionTuple versionTuple;
            if (!versionStr.empty()) {
                if (versionTuple.tryParse(versionStr)) {
                    errorHandler.AddError("Invalid version format: \"%s\". Use <major_ver.minor_ver>.", version);
                }
            } else {
                versionTuple = darwinUnspecifiedVersion;
            }
            g->darwinVersionMin = versionTuple;
#endif
        } else if (!strcmp(argv[i], "--print-target")) {
            g->printTarget = true;
        } else if (!strcmp(argv[i], "--no-omit-frame-pointer")) {
            g->NoOmitFramePointer = true;
        } else if (!strcmp(argv[i], "--instrument")) {
            g->emitInstrumentation = true;
        } else if (!strcmp(argv[i], "--no-pragma-once")) {
            g->noPragmaOnce = true;
        } else if (!strcmp(argv[i], "-g")) {
            g->generateDebuggingSymbols = true;
        } else if (!strcmp(argv[i], "--sample-profiling-debug-info")) {
            g->sampleProfilingDebugInfo = true;
        } else if (!strncmp(argv[i], "--profile-sample-use=", 21)) {
            g->profileSampleUse = argv[i] + 21;
        } else if (!strcmp(argv[i], "-E")) {
            g->onlyCPP = true;
            output.type = Module::CPPStub;
            // g->preprocessorOutputType is initialized as "Cpp" automatically
        } else if (!strcmp(argv[i], "-dM")) {
            g->onlyCPP = true;
            output.type = Module::CPPStub;
            g->preprocessorOutputType = Globals::PreprocessorOutputType::MacrosOnly;
        } else if (!strcmp(argv[i], "-dD")) {
            g->onlyCPP = true;
            output.type = Module::CPPStub;
            g->preprocessorOutputType = Globals::PreprocessorOutputType::WithMacros;
        } else if (!strcmp(argv[i], "--emit-asm")) {
            output.type = Module::Asm;
        } else if (!strcmp(argv[i], "--emit-llvm")) {
            output.type = Module::Bitcode;
        } else if (!strcmp(argv[i], "--emit-llvm-text")) {
            output.type = Module::BitcodeText;
        } else if (!strcmp(argv[i], "--emit-obj")) {
            output.type = Module::Object;
        }
#ifdef ISPC_XE_ENABLED
        else if (!strcmp(argv[i], "--emit-spirv")) {
            output.type = Module::SPIRV;
        } else if (!strcmp(argv[i], "--emit-zebin")) {
            output.type = Module::ZEBIN;
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
        } else if (!strncmp(argv[i], "-I", 2)) {
            lParseInclude(argv[i] + 2);
        } else if (!strcmp(argv[i], "--ignore-preprocessor-errors")) {
            g->ignoreCPPErrors = true;
        } else if (!strcmp(argv[i], "--target")) {
            // FIXME: should remove this way of specifying the target...
            if (++i != argc) {
                auto result = ParseISPCTargets(argv[i]);
                targets = result.first;
                if (!result.second.empty()) {
                    errorHandler.AddError("Incorrect targets: %s.  Choices are: %s. Use --support-matrix for complete "
                                          "information about supported targets, archs and target OSes.",
                                          result.second.c_str(), g->target_registry->getSupportedTargets().c_str());
                }
            } else {
                errorHandler.AddError("No target specified after --target option.");
            }
        } else if (!strncmp(argv[i], "--target=", 9)) {
            auto result = ParseISPCTargets(argv[i] + 9);
            targets = result.first;
            if (!result.second.empty()) {
                errorHandler.AddError("Incorrect targets: %s.  Choices are: %s. Use --support-matrix for complete "
                                      "information about supported targets, archs and target OSes.",
                                      result.second.c_str(), g->target_registry->getSupportedTargets().c_str());
            }
        } else if (!strncmp(argv[i], "--target-os=", 12)) {
            g->target_os = ParseOS(argv[i] + 12);
            if (g->target_os == TargetOS::error) {
                errorHandler.AddError(
                    "Unsupported value for --target-os, supported values are: %s. Use --support-matrix for complete "
                    "information about supported targets, archs and target OSes.",
                    g->target_registry->getSupportedOSes().c_str());
            }
        } else if (!strcmp(argv[i], "--no-vectorcall")) {
            vectorCall = BooleanOptValue::disabled;
        } else if (!strcmp(argv[i], "--vectorcall")) {
            vectorCall = BooleanOptValue::enabled;
        } else if (!strncmp(argv[i], "--math-lib=", 11)) {
            const char *lib = argv[i] + 11;
            if (!strcmp(lib, "default")) {
                g->mathLib = Globals::MathLib::Math_ISPC;
            } else if (!strcmp(lib, "fast")) {
                g->mathLib = Globals::MathLib::Math_ISPCFast;
            } else if (!strcmp(lib, "svml")) {
                g->mathLib = Globals::MathLib::Math_SVML;
            } else if (!strcmp(lib, "system")) {
                g->mathLib = Globals::MathLib::Math_System;
            } else {
                errorHandler.AddError("Unknown --math-lib= option \"%s\".", lib);
            }
        } else if (!strncmp(argv[i], "--opt=", 6)) {
            const char *opt = argv[i] + 6;
            if (!strcmp(opt, "fast-math")) {
                g->opt.fastMath = true;
            } else if (!strcmp(opt, "fast-masked-vload")) {
                g->opt.fastMaskedVload = true;
            } else if (!strcmp(opt, "disable-assertions")) {
                g->opt.disableAsserts = true;
            } else if (!strcmp(opt, "disable-gathers")) {
                g->opt.disableGathers = true;
            } else if (!strcmp(opt, "disable-scatters")) {
                g->opt.disableScatters = true;
            } else if (!strcmp(opt, "disable-loop-unroll")) {
                g->opt.unrollLoops = false;
            } else if (!strcmp(opt, "disable-fma")) {
                g->opt.disableFMA = true;
            } else if (!strcmp(opt, "disable-zmm")) {
                g->opt.disableZMM = true;
            } else if (!strcmp(opt, "force-aligned-memory")) {
                g->opt.forceAlignedMemory = true;
            } else if (!strcmp(opt, "reset-ftz-daz")) {
                g->opt.resetFTZ_DAZ = true;
            }

            // These are only used for performance tests of specific
            // optimizations
            else if (!strcmp(opt, "enable-ldst-vectorizer")) {
                g->opt.enableLoadStoreVectorizer = true;
            } else if (!strcmp(opt, "enable-slp-vectorizer")) {
                g->opt.enableSLPVectorizer = true;
            } else if (!strcmp(opt, "disable-all-on-optimizations")) {
                g->opt.disableMaskAllOnOptimizations = true;
            } else if (!strcmp(opt, "disable-coalescing")) {
                g->opt.disableCoalescing = true;
            } else if (!strcmp(opt, "disable-handle-pseudo-memory-ops")) {
                g->opt.disableHandlePseudoMemoryOps = true;
            } else if (!strcmp(opt, "disable-blended-masked-stores")) {
                g->opt.disableBlendedMaskedStores = true;
            } else if (!strcmp(opt, "disable-coherent-control-flow")) {
                g->opt.disableCoherentControlFlow = true;
            } else if (!strcmp(opt, "disable-uniform-control-flow")) {
                g->opt.disableUniformControlFlow = true;
            } else if (!strcmp(opt, "disable-gather-scatter-optimizations")) {
                g->opt.disableGatherScatterOptimizations = true;
            } else if (!strcmp(opt, "disable-blending-removal")) {
                g->opt.disableMaskedStoreToStore = true;
            } else if (!strcmp(opt, "disable-gather-scatter-flattening")) {
                g->opt.disableGatherScatterFlattening = true;
            } else if (!strcmp(opt, "disable-uniform-memory-optimizations")) {
                g->opt.disableUniformMemoryOptimizations = true;
            }
#ifdef ISPC_XE_ENABLED
            else if (!strcmp(opt, "disable-xe-gather-coalescing")) {
                g->opt.disableXeGatherCoalescing = true;
            } else if (!strncmp(opt, "threshold-for-xe-gather-coalescing=", 37)) {
                g->opt.thresholdForXeGatherCoalescing = atoi(opt + 37);
            } else if (!strcmp(opt, "emit-xe-hardware-mask")) {
                g->opt.emitXeHardwareMask = true;
            } else if (!strcmp(opt, "enable-xe-foreach-varying")) {
                g->opt.enableForeachInsideVarying = true;
            } else if (!strcmp(opt, "enable-xe-unsafe-masked-load")) {
                g->opt.enableXeUnsafeMaskedLoad = true;
            }
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
        } else if (!strcmp(argv[i], "--werror")) {
            g->warningsAsErrors = true;
        } else if (!strcmp(argv[i], "--wrap-signed-int")) {
            wrapSignedInt = BooleanOptValue::enabled;
        } else if (!strcmp(argv[i], "--no-wrap-signed-int")) {
            wrapSignedInt = BooleanOptValue::disabled;
        } else if (!strncmp(argv[i], "--error-limit=", 14)) {
            int errLimit = atoi(argv[i] + 14);
            if (errLimit >= 0) {
                g->errorLimit = errLimit;
            } else {
                errorHandler.AddError("Invalid value for --error-limit: \"%d\" -- "
                                      "value cannot be a negative number.",
                                      errLimit);
            }
        } else if (!strcmp(argv[i], "--nowrap")) {
            g->disableLineWrap = true;
        } else if (!strcmp(argv[i], "--wno-perf") || !strcmp(argv[i], "-wno-perf")) {
            g->emitPerfWarnings = false;
        } else if (!strcmp(argv[i], "--stack-protector")) {
            g->SSPLevel = SSPKind::SSPOn;
        } else if (!strncmp(argv[i], "--stack-protector=", 18)) {
            const char *level = argv[i] + strlen("--stack-protector=");
            if (!strcmp(level, "all")) {
                g->SSPLevel = SSPKind::SSPReq;
            } else if (!strcmp(level, "on")) {
                g->SSPLevel = SSPKind::SSPOn;
            } else if (!strcmp(level, "none")) {
                g->SSPLevel = SSPKind::SSPNone;
            } else if (!strcmp(level, "strong")) {
                g->SSPLevel = SSPKind::SSPStrong;
            } else {
                errorHandler.AddError("Unknown --stack-protector= option \"%s\".", level);
            }
        } else if (!strcmp(argv[i], "-o")) {
            if (++i != argc) {
                output.out = argv[i];
            } else {
                errorHandler.AddError("No output file specified after -o option.");
            }
        } else if (!strncmp(argv[i], "--outfile=", 10)) {
            output.out = argv[i] + strlen("--outfile=");
        } else if (!strcmp(argv[i], "-h")) {
            if (++i != argc) {
                output.header = argv[i];
            } else {
                errorHandler.AddError("No header file name specified after -h option.");
            }
        } else if (!strncmp(argv[i], "--header-outfile=", 17)) {
            output.header = argv[i] + strlen("--header-outfile=");
        } else if (!strncmp(argv[i], "--nanobind-wrapper=", 19)) {
            output.nbWrap = argv[i] + strlen("--nanobind-wrapper=");
        } else if (!strcmp(argv[i], "-O0")) {
            g->opt.level = 0;
            g->codegenOptLevel = Globals::CodegenOptLevel::None;
        } else if (!strcmp(argv[i], "-O1")) {
            g->opt.level = 1;
            g->codegenOptLevel = Globals::CodegenOptLevel::Default;
            g->opt.disableCoherentControlFlow = true;
        } else if (!strcmp(argv[i], "-O") || !strcmp(argv[i], "-O2") || !strcmp(argv[i], "-O3")) {
            g->opt.level = 2;
            g->codegenOptLevel = Globals::CodegenOptLevel::Aggressive;
        } else if (!strcmp(argv[i], "-")) {
            file = argv[i];
        } else if (!strcmp(argv[i], "--nostdlib")) {
            g->includeStdlib = false;
        } else if (!strcmp(argv[i], "--include-float16-conversions")) {
            g->includeFloat16Conversions = true;
        } else if (!strcmp(argv[i], "--nocpp")) {
            g->runCPP = false;
            Warning(SourcePos(), "--nocpp is deprecated and will be removed in the future. ");
        } else if (!strcmp(argv[i], "-ffunction-sections")) {
            g->functionSections = true;
        } else if (!strcmp(argv[i], "-fno-function-sections")) {
            g->functionSections = false;
        } else if (!strncmp(argv[i], "--mcmodel=", 10)) {
            const char *value = argv[i] + 10;
            if (!strcmp(value, "small")) {
                output.flags.setMCModel(MCModel::Small);
            } else if (!strcmp(value, "large")) {
                output.flags.setMCModel(MCModel::Large);
            } else {
                errorHandler.AddError("Unsupported code model \"%s\". Only small and large models are supported.",
                                      value);
            }
        } else if (!strcmp(argv[i], "--pic")) {
            output.flags.setPICLevel(PICLevel::SmallPIC);
        } else if (!strcmp(argv[i], "--PIC")) {
            output.flags.setPICLevel(PICLevel::BigPIC);
        }
#ifndef ISPC_IS_HOST_WINDOWS
        else if (!strcmp(argv[i], "--colored-output")) {
            g->forceColoredOutput = true;
        }
#endif // !ISPC_IS_HOST_WINDOWS
        else if (!strcmp(argv[i], "--quiet")) {
            g->quiet = true;
        } else if (!strcmp(argv[i], "--yydebug")) {
            yydebug = 1;
        } else if (!strcmp(argv[i], "-MMM")) {
            if (++i != argc) {
                output.deps = argv[i];
                output.flags.setFlatDeps();
            } else {
                errorHandler.AddError("No output file name specified after -MMM option.");
            }
        } else if (!strcmp(argv[i], "-M")) {
            output.flags.setMakeRuleDeps();
            output.flags.setDepsToStdout();
        } else if (!strcmp(argv[i], "-MF")) {
            if (++i != argc) {
                output.deps = argv[i];
            } else {
                errorHandler.AddError("No output file name specified after -MF option.");
            }
        } else if (!strcmp(argv[i], "-MT")) {
            if (++i != argc) {
                output.depsTarget = argv[i];
            } else {
                errorHandler.AddError("No target name specified after -MT option.");
            }
        } else if (!strcmp(argv[i], "--dev-stub")) {
            if (++i != argc) {
                output.devStub = argv[i];
            } else {
                errorHandler.AddError("No output file name specified after --dev-stub option.");
            }
        } else if (!strcmp(argv[i], "--host-stub")) {
            if (++i != argc) {
                output.hostStub = argv[i];
            } else {
                errorHandler.AddError("No output file name specified after --host-stub option.");
            }
        } else if (strncmp(argv[i], "--debug-phase=", 14) == 0) {
            errorHandler.AddWarning("Adding debug phases may change the way PassManager"
                                    "handles the phases and it may possibly make some bugs go"
                                    "away or introduce the new ones.");
            g->debug_stages = lParsingPhases(argv[i] + strlen("--debug-phase="), errorHandler);
        } else if (strncmp(argv[i], "--dump-file=", 12) == 0) {
            g->dumpFile = true;
            g->dumpFilePath = lParsePath(argv[i] + strlen("--dump-file="), errorHandler);
        } else if (strncmp(argv[i], "--dump-file", 11) == 0) {
            g->dumpFile = true;
        } else if (strncmp(argv[i], "--gen-stdlib", 12) == 0) {
            g->genStdlib = true;
        } else if (strncmp(argv[i], "--check-bitcode-libs", 20) == 0) {
            for (auto &name : g->target_registry->checkBitcodeLibs()) {
                errorHandler.AddError("Missed builtins/stdlib library: \"%s\"", name.c_str());
            }
            errorHandler.Emit();
            return ArgsParseResult::help_requested;
        } else if (strncmp(argv[i], "--off-phase=", 12) == 0) {
            g->off_stages = lParsingPhases(argv[i] + strlen("--off-phase="), errorHandler);
#ifdef ISPC_XE_ENABLED
        } else if (!strncmp(argv[i], "--vc-options=", 13)) {
            g->vcOpts = argv[i] + strlen("--vc-options=");
        } else if (!strncmp(argv[i], "--xe-stack-mem-size=", 20)) {
            unsigned int memSize = atoi(argv[i] + 20);
            g->stackMemSize = memSize;
#endif
        } else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--version")) {
            lPrintVersion();
            return ArgsParseResult::help_requested;
        } else if (argv[i][0] == '-') {
            errorHandler.AddError("Unknown option \"%s\".", argv[i]);
        } else {
            if (!file.empty()) {
                errorHandler.AddError("Multiple input files specified on command "
                                      "line: \"%s\" and \"%s\".",
                                      file.c_str(), argv[i]);
            } else {
                file = argv[i];
            }
        }
    }

    // Emit accumulted errors and warnings, if any.
    // All the rest of errors and warnigns will be processed in regullar way.
    if (auto result = errorHandler.Emit(); result != ArgsParseResult::success) {
        return result;
    }

    if (g->genStdlib) {
        std::string stdlib = "stdlib/stdlib.ispc";
        std::string generic = "builtins/generic.ispc";
        if (stdlib != file && generic != file) {
            Error(SourcePos(), "The --gen-stdlib option can be used only with stdlib.ispc or builtins/generic.ispc.");
            return ArgsParseResult::failure;
        }
    }

    // If [no]discard-value-names is explicitly specified, then use this value.
    // Otherwise enable it only if the output is some form of bitcode.
    // Note that discarding value names significantly saves compile time and memory consumption.
    if (discardValueNames == BooleanOptValue::enabled) {
        g->ctx->setDiscardValueNames(true);
    } else if (discardValueNames == BooleanOptValue::disabled) {
        g->ctx->setDiscardValueNames(false);
    } else {
        if (output.type == Module::Bitcode || output.type == Module::BitcodeText) {
            g->ctx->setDiscardValueNames(false);
        } else {
            g->ctx->setDiscardValueNames(true);
        }
    }

    // Default settings for PS4
    if (g->target_os == TargetOS::ps4 || g->target_os == TargetOS::ps5) {
        output.flags.setPICLevel(PICLevel::BigPIC);
        if (cpu.empty()) {
            if (g->target_os == TargetOS::ps4) {
                // Default for PS4 is btver2, but do not enforce it.
                cpu = "btver2";
            } else {
                // Default for PS5 is znver2, but do not enforce it.
                cpu = "znver2";
            }
        }
        if (arch != Arch::x86_64) {
            Warning(SourcePos(), "--arch switch is ignored for PS4/PS5 target OS. x86-64 arch is used.");
            arch = Arch::x86_64;
        }
    }

    // Default setting for "custom_linux"
    if (g->target_os == TargetOS::custom_linux) {
        output.flags.setPICLevel();
        if (cpu.empty()) {
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

#ifdef ISPC_WASM_ENABLED
    // Default setting for wasm
    if (arch == Arch::wasm32 || arch == Arch::wasm64) {
        g->target_os = TargetOS::web;
    }
    for (auto target : targets) {
        if (target == ISPCTarget::wasm_i32x4) {
            Assert(targets.size() == 1 && "wasm supports only one target: i32x4");
            g->target_os = TargetOS::web;
            if (arch == Arch::none) {
                arch = Arch::wasm32;
            }
        }
    }
    if (g->target_os == TargetOS::web) {
        if (arch == Arch::none) {
            arch = Arch::wasm32;
        }
        if (targets.empty()) {
            targets.push_back(ISPCTarget::wasm_i32x4);
        }
    }
#endif

    // Validate mcmodel and architecture combination
    if (output.flags.getMCModel() == MCModel::Large && arch == Arch::x86) {
        Error(SourcePos(), "--mcmodel=large is not supported for x86 architecture. "
                           "Use x86-64 architecture instead.");
        return ArgsParseResult::failure;
    }

    if (!output.deps.empty()) {
        output.flags.setDepsToStdout(false);
    }

    if (!output.deps.empty() && !output.flags.isFlatDeps() && !output.flags.isMakeRuleDeps()) {
        Warning(SourcePos(), "Dependency file name specified with -MF, but no "
                             "mode specified; did you forget to specify -M or -MMM? "
                             "No dependency output will be generated.");
        output.deps.clear();
    }

    if (output.flags.isFlatDeps() && output.flags.isMakeRuleDeps()) {
        Warning(SourcePos(), "Both -M and -MMM specified on the command line. "
                             "-MMM takes precedence.");
        output.flags.setFlatDeps(false);
    }

    if (g->onlyCPP && output.out.empty()) {
        output.out = "-"; // Assume stdout by default (-E mode)
    }

    if (g->target_os == TargetOS::windows && output.flags.isPIC()) {
        Warning(SourcePos(), "--pic|--PIC switches for Windows target will be ignored.");
    }

    if (g->target_os != TargetOS::windows && g->dllExport) {
        Warning(SourcePos(), "--dllexport switch will be ignored, as the target OS is not Windows.");
    }

    if (vectorCall != BooleanOptValue::none &&
        (g->target_os != TargetOS::windows ||
         // This is a hacky check. Arch is properly set later, so we rely that default means x86_64.
         (arch != Arch::x86_64 && arch != Arch::none))) {
        Warning(SourcePos(), "--vectorcall/--no-vectorcall are supported only for x86_64 Windows target, so these "
                             "options will be ignored.");
    }

    if (targets.size() > 1) {
        g->isMultiTargetCompilation = true;
    }

    // Validate --opt=disable-zmm usage
    if (g->opt.disableZMM) {
        for (auto target : targets) {
            if (target != ISPCTarget::avx512skx_x16 && target != ISPCTarget::avx512icl_x16 &&
                target != ISPCTarget::generic_i32x16 && target != ISPCTarget::generic_i1x16 &&
                target != ISPCTarget::generic_i1x32 && target != ISPCTarget::generic_i1x64) {
                Warning(SourcePos(),
                        "--opt=disable-zmm can only be used with avx512skx-x16, avx512icl-x16 or generic-i32x16, "
                        "generic-i1x16, generic-i1x32, generic-i1x64 targets.");
            }
        }
    }

    if ((output.type == Module::Asm) && (intelAsmSyntax != nullptr)) {
        std::vector<const char *> Args(3);
        Args[0] = "ispc (LLVM option parsing)";
        Args[2] = nullptr;
        if (std::string(intelAsmSyntax) == "intel") {
            Args[1] = "--x86-asm-syntax=intel";
        } else {
            Args[1] = "--x86-asm-syntax=att";
        }
        llvm::cl::ParseCommandLineOptions(2, Args.data());
    }

    bool targetIsGen = false;
#ifdef ISPC_XE_ENABLED
    for (auto target : targets) {
        if (ISPCTargetIsGen(target)) {
            targetIsGen = true;
            Assert(targets.size() == 1 && "multi-target is not supported for Xe targets yet.");
            // Generate .spv for Xe target instead of object by default.
            if (output.type == Module::Object) {
                Warning(SourcePos(), "Emitting spir-v file for Xe targets.");
                output.type = Module::SPIRV;
            }
        }

        if (target == ISPCTarget::gen9_x8 || target == ISPCTarget::gen9_x16) {
            Warning(SourcePos(), "The target %s is deprecated and will be removed in the future.",
                    ISPCTargetToString(target).c_str());
        }
    }
#endif

    // If [no]wrap-signed-int is explicitly specified, then use this value.
    // Disable NSW bit optimization by default due to performance regressions
    // on some GPU workloads.  Otherwise enable it by default only for CPU targets.
    if (wrapSignedInt == BooleanOptValue::enabled) {
        g->wrapSignedInt = true;
    } else if (wrapSignedInt == BooleanOptValue::disabled) {
        g->wrapSignedInt = false;
    } else if (targetIsGen) {
        g->wrapSignedInt = true;
    } else {
        g->wrapSignedInt = false;
    }

    // This needs to happen after the TargetOS is decided.
    setCallingConv(vectorCall, arch);
    return ArgsParseResult::success;
}

bool ispc::ValidateInput(const std::string &filename, bool allowStdin) {
    if (filename.empty()) {
        if (allowStdin) {
            Error(SourcePos(), "No input file were specified. To read text from stdin use \"-\" as file name.");
        } else {
            Error(SourcePos(), "No input file specified.");
        }
        return false;
    }

    if (filename != "-") {
        // If the input is not stdin then check that the file exists and it is
        // not a directory.
        if (!llvm::sys::fs::exists(filename)) {
            Error(SourcePos(), "File \"%s\" does not exist.", filename.c_str());
            return false;
        }

        if (llvm::sys::fs::is_directory(filename)) {
            Error(SourcePos(), "File \"%s\" is a directory.", filename.c_str());
            return false;
        }
    }
    return true;
}

void ispc::ValidateOutput(const Module::Output &output) {
    if (output.out.empty() && output.header.empty() && (output.deps.empty() && !output.flags.isDepsToStdout()) &&
        output.hostStub.empty() && output.devStub.empty() && output.nbWrap.empty()) {
        Warning(SourcePos(), "No output file or header file name specified. "
                             "Program will be compiled and warnings/errors will "
                             "be issued, but no output will be generated.");
    }
}
