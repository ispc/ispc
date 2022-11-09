/*
  Copyright (c) 2022, Intel Corporation
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

#include "DemotePHIsPass.h"

#ifdef ISPC_XE_ENABLED
namespace ispc {

char DemotePHIs::ID = 0;

bool DemotePHIs::runOnFunction(llvm::Function &F) {
    llvm::TimeTraceScope FuncScope("DemotePHIs::runOnFunction", F.getName());
    if (F.isDeclaration() || skipFunction(F))
        return false;
    std::vector<llvm::Instruction *> WorkList;
    for (auto &ibb : F)
        for (llvm::BasicBlock::iterator iib = ibb.begin(), iie = ibb.end(); iib != iie; ++iib)
            if (llvm::isa<llvm::PHINode>(iib))
                WorkList.push_back(&*iib);

    // Demote phi nodes
    for (auto *ilb : llvm::reverse(WorkList))
        DemotePHIToStack(llvm::cast<llvm::PHINode>(ilb), nullptr);

    return !WorkList.empty();
}

llvm::Pass *CreateDemotePHIs() { return new DemotePHIs(); }
} // namespace ispc
#endif