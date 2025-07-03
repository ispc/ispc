/*
  Copyright (c) 2010-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file main.cpp
    @brief main() entrypoint implementation for ispc
*/

#include "args.h"
#include "binary_type.h"
#include "ispc/compiler.h"
#include "util.h"
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Signals.h>

#ifdef ISPC_HOST_IS_WINDOWS
#include <windows.h>
#else
#include <unistd.h>
#endif // ISPC_HOST_IS_WINDOWS

using namespace ispc;

static void lSignal(void *) { FATAL("Unhandled signal sent to process; terminating."); }

int main(int Argc, char *Argv[]) {
#ifdef ISPC_HOST_IS_WINDOWS
    // Before any LLVM calls, remove CWD from the DLL search path to reduce the risk of DLL injection.
    // https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order
    SetDllDirectory("");
#endif
    llvm::sys::AddSignalHandler(lSignal, nullptr);

    // Process potential @args files
    std::vector<char *> argv;
    GetAllArgs(Argc, Argv, argv);
    int argc = argv.size();

    int ret = 1; // Default to failure

    // Initialize the ISPC library
    if (!ispc::Initialize()) {
        FreeArgv(argv);
        return ret;
    }

    std::string ISPCAbsPath = llvm::sys::fs::getMainExecutable(argv[0], (void *)(intptr_t)main);
    initializeBinaryType(ISPCAbsPath.c_str());
    
    // Execute the compilation process
    ret = ispc::CompileFromArgs(argc, argv.data());

    // Perform final global cleanup
    ispc::Shutdown();
    FreeArgv(argv);

    return ret;
}