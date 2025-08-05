/*
  Copyright (c) 2010-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file main.cpp
    @brief main() entrypoint implementation for ispc
*/

#include "args.h"
#include "binary_type.h"
#include "ispc/ispc.h"
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

static void lSignal(void *) { FATAL("Unhandled signal sent to process; terminating."); }

int main(int Argc, char *Argv[]) {
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

    std::vector<char *> argv;
    GetAllArgs(Argc, Argv, argv);

    int ret = 1; // Default to failure

    // Initialize the ISPC engine
    if (!ispc::Initialize()) {
        FreeArgv(argv);
        return ret;
    }

    std::string ISPCAbsPath = llvm::sys::fs::getMainExecutable(argv[0], (void *)(intptr_t)main);
    initializeBinaryType(ISPCAbsPath.c_str());

    // Execute the compilation process using C-style interface
    // Skip the program name (argv[0]) since the interface no longer requires it
    ret = ispc::CompileFromCArgs(argv.size() - 1, argv.data() + 1);

    // Perform final global cleanup
    ispc::Shutdown();
    FreeArgv(argv);

    return ret;
}