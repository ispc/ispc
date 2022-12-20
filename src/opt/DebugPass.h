/*
  Copyright (c) 2022-2023, Intel Corporation
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
