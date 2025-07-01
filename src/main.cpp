/*
  Copyright (c) 2010-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file main.cpp
    @brief main() entrypoint implementation for ispc
*/

#include "args.h"
#include "binary_type.h"
#include "ispc.h"
#include "target_registry.h"
#include "type.h"
#include "util.h"

#include <cstdarg>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#ifdef ISPC_HOST_IS_WINDOWS
#include <time.h>
#include <windows.h>
#else
#include <unistd.h>
#endif // ISPC_HOST_IS_WINDOWS

#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ToolOutputFile.h>

using namespace ispc;

static int toExitCode(ArgsParseResult code) {
    switch (code) {
    case ArgsParseResult::success:
    case ArgsParseResult::help_requested:
        return 0;
    case ArgsParseResult::failure:
        return 1;
    }
    return 1; // Default to failure
}

static void lSignal(void *) { FATAL("Unhandled signal sent to process; terminating."); }

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

int main(int Argc, char *Argv[]) {
    std::vector<char *> argv;
    GetAllArgs(Argc, Argv, argv);
    int argc = argv.size();
#ifdef ISPC_HOST_IS_WINDOWS
    // While ispc doesn't load any libraries explicitly using LoadLibrary API (or alternatives), it uses vcruntime that
    // loads vcruntime140.dll and msvcp140.dll. Moreover LLVM loads dbghelp.dll.
    // There is no way to modify DLL search order for vcruntime140.dll and msvcp140.dll but we
    // can prevent searching in CWD while loading dbghelp.dll.
    // So before initiating any LLVM call, remove CWD from the search path to reduce the risk of DLL injection
    // when Safe DLL search mode is OFF.
    // https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order
    SetDllDirectory("");
#endif
    llvm::sys::AddSignalHandler(lSignal, nullptr);
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
    char *file = nullptr;

    std::vector<std::string> linkFileNames;
    // Initiailize globals early so that we can set various option values
    // as we're parsing below
    g = new Globals;

    Module::Output output;
    output.type = Module::Object; // Default output type
    Arch arch = Arch::none;
    std::vector<ISPCTarget> targets;
    const char *cpu = nullptr;

    std::string ISPCAbsPath = llvm::sys::fs::getMainExecutable(argv[0], (void *)(intptr_t)main);
    initializeBinaryType(ISPCAbsPath.c_str());

    // Parse command line options
    bool isLinkMode = false;
    ArgsParseResult parseResult =
        ParseCommandLineArgs(argc, argv.data(), file, arch, cpu, targets, output, linkFileNames, isLinkMode);

    if (parseResult != ArgsParseResult::success) {
        // Print help and exit.
        FreeArgv(argv);
        delete g;
        return toExitCode(parseResult);
    }

    int ret = 0;

    if (isLinkMode) {
        // Handle link mode
        std::string filename = !output.out.empty() ? output.out : "";
        ret = Module::LinkAndOutput(linkFileNames, output.type, filename);
    } else {
        if (g->enableTimeTrace) {
            llvm::timeTraceProfilerInitialize(g->timeTraceGranularity, "ispc");
        }
        {
            llvm::TimeTraceScope TimeScope("ExecuteCompiler");
            ret = Module::CompileAndOutput(file, arch, cpu, targets, output);
        }

        if (g->enableTimeTrace) {
            // Write to file only if compilation is successfull.
            if ((ret == 0) && (!output.out.empty())) {
                writeCompileTimeFile(output.out.c_str());
            }
            llvm::timeTraceProfilerCleanup();
        }
    }

    // Free all bookkeeped objects.
    BookKeeper::in().freeAll();

    FreeArgv(argv);
    return ret;
}