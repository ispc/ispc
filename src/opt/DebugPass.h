/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

#include <sstream>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Regex.h>

namespace ispc {

/** This pass is added in list of passes after optimizations which
    we want to debug and print dump of LLVM IR in stderr. Also it
    prints name and number of previous optimization.
 */
class DebugPass : public llvm::PassInfoMixin<DebugPass> {
  public:
    explicit DebugPass(char *output) { snprintf(str_output, sizeof(str_output), "%s", output); }

    static llvm::StringRef getPassName() { return "Dump LLVM IR"; }
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

  private:
    char str_output[100];
};

/** This pass is added in list of passes after optimizations which
    we want to debug and print dump of LLVM IR to file.
 */
class DebugPassFile : public llvm::PassInfoMixin<DebugPassFile> {
  public:
    explicit DebugPassFile(int number, llvm::StringRef name, std::string dir) : pnum(number), pname(name), pdir(dir) {}

    static llvm::StringRef getPassName() { return "Dump LLVM IR"; }
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

  private:
    void run(llvm::Module &m, bool init);
    int pnum;
    llvm::StringRef pname;
    std::string pdir;
};

} // namespace ispc
