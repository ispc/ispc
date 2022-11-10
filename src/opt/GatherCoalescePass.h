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

#pragma once

#include "ISPCPass.h"

namespace ispc {

// This pass implements two optimizations to improve the performance of
// gathers; currently only gathers of 32-bit values where it can be
// determined at compile time that the mask is all on are supported, though
// both of those limitations may be generalized in the future.
//
//  First, for any single gather, see if it's worthwhile to break it into
//  any of scalar, 2-wide (i.e. 64-bit), 4-wide, or 8-wide loads.  Further,
//  we generate code that shuffles these loads around.  Doing fewer, larger
//  loads in this manner, when possible, can be more efficient.
//
//  Second, this pass can coalesce memory accesses across multiple
//  gathers. If we have a series of gathers without any memory writes in
//  the middle, then we try to analyze their reads collectively and choose
//  an efficient set of loads for them.  Not only does this help if
//  different gathers reuse values from the same location in memory, but
//  it's specifically helpful when data with AOS layout is being accessed;
//  in this case, we're often able to generate wide vector loads and
//  appropriate shuffles automatically.

class GatherCoalescePass : public llvm::FunctionPass {
  public:
    static char ID;
    explicit GatherCoalescePass() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const override { return "Gather Coalescing"; }
    bool runOnFunction(llvm::Function &F) override;

  private:
    // Type of base pointer element type (the 1st argument of the intrinsic) is i8
    // e.g. @__pseudo_gather_factored_base_offsets32_i32(i8 *, <WIDTH x i32>, i32, <WIDTH x i32>, <WIDTH x MASK>)
    llvm::Type *baseType{LLVMTypes::Int8Type};
    bool coalesceGathersFactored(llvm::BasicBlock &BB);
};

llvm::Pass *CreateGatherCoalescePass();

} // namespace ispc