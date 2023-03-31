/*
  Copyright (c) 2022-2023, Intel Corporation
  All rights reserved.

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
class DebugPass : public llvm::ModulePass {
  public:
    static char ID;
    explicit DebugPass(char *output) : ModulePass(ID) { snprintf(str_output, sizeof(str_output), "%s", output); }

    llvm::StringRef getPassName() const override { return "Dump LLVM IR"; }
    bool runOnModule(llvm::Module &m) override;

  private:
    char str_output[100];
};
llvm::Pass *CreateDebugPass(char *output);

/** This pass is added in list of passes after optimizations which
    we want to debug and print dump of LLVM IR to file.
 */
class DebugPassFile : public llvm::ModulePass {
  public:
    static char ID;
    explicit DebugPassFile(int number, llvm::StringRef name, std::string dir)
        : ModulePass(ID), pnum(number), pname(name), pdir(dir) {}

    llvm::StringRef getPassName() const override { return "Dump LLVM IR"; }
    bool runOnModule(llvm::Module &m) override;
    bool doInitialization(llvm::Module &m) override;

  private:
    void run(llvm::Module &m, bool init);
    int pnum;
    llvm::StringRef pname;
    std::string pdir;
};

llvm::Pass *CreateDebugPassFile(int number, llvm::StringRef name, std::string dir);
} // namespace ispc
