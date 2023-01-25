/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "DebugPass.h"

namespace ispc {

llvm::PreservedAnalyses DebugPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    fprintf(stderr, "%s", str_output);
    fflush(stderr);
    M.print(llvm::errs(), nullptr);
    return llvm::PreservedAnalyses::all();
}

/**
 * Strips all non-alphanumeric characters from given string.
 */
static std::string lSanitize(std::string in) {
    llvm::Regex r("[^[:alnum:]]");
    while (r.match(in))
        in = r.sub("", in);
    return in;
}

void DebugPassFile::run(llvm::Module &module, bool init) {
    std::ostringstream oss;
    oss << (init ? "init_" : "ir_") << pnum << "_" << lSanitize(std::string{pname}) << ".ll";

    const std::string pathFile{oss.str()};

#ifdef ISPC_HOST_IS_WINDOWS
    const std::string pathSep{"\\"};
#else
    const std::string pathSep{"/"};
#endif // ISPC_HOST_IS_WINDOWS

    std::string pathDirFile;

    if (!pdir.empty()) {
        llvm::sys::fs::create_directories(pdir);
        pathDirFile = pdir + pathSep + pathFile;
    } else {
        pathDirFile = pathFile;
    }

    std::error_code EC;
    llvm::raw_fd_ostream OS(pathDirFile, EC, llvm::sys::fs::OF_None);
    Assert(!EC && "IR dump file creation failed!");
    module.print(OS, 0);
}

llvm::PreservedAnalyses DebugPassFile::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    run(M, false);
    return llvm::PreservedAnalyses::all();
}

} // namespace ispc
