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

/** @file ISPCPass.h
    @brief Header file with declarations common for ISPC passes.
*/

#pragma once

#include "ispc.h"
#include "llvmutil.h"
#include "module.h"
#include "util.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/IR/Verifier.h>
#include <llvm/InitializePasses.h>
#include <llvm/Pass.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/FunctionAttrs.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Vectorize.h>

#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXIntrOpts.h>
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
// Used for Xe gather coalescing
#include <llvm/Transforms/Utils/Local.h>
#endif

#ifndef PRId64
#define PRId64 "lld"
#endif
#ifndef PRIu64
#define PRIu64 "llu"
#endif

namespace ispc {

// Constant in number of bytes.
enum { BYTE = 1, WORD = 2, DWORD = 4, QWORD = 8, OWORD = 16, GRF = 32 };

#define DEBUG_START_BB(NAME)                                                                                           \
    if (g->debugPrint &&                                                                                               \
        (getenv("FUNC") == NULL || (getenv("FUNC") != NULL && !strncmp(bb.getParent()->getName().str().c_str(),        \
                                                                       getenv("FUNC"), strlen(getenv("FUNC")))))) {    \
        fprintf(stderr, "Start of " NAME "\n");                                                                        \
        fprintf(stderr, "---------------\n");                                                                          \
        bb.print(llvm::errs());                                                                                        \
        fprintf(stderr, "---------------\n\n");                                                                        \
    } else /* eat semicolon */

#define DEBUG_END_BB(NAME)                                                                                             \
    if (g->debugPrint &&                                                                                               \
        (getenv("FUNC") == NULL || (getenv("FUNC") != NULL && !strncmp(bb.getParent()->getName().str().c_str(),        \
                                                                       getenv("FUNC"), strlen(getenv("FUNC")))))) {    \
        fprintf(stderr, "End of " NAME " %s\n", modifiedAny ? "** CHANGES **" : "");                                   \
        fprintf(stderr, "---------------\n");                                                                          \
        bb.print(llvm::errs());                                                                                        \
        fprintf(stderr, "---------------\n\n");                                                                        \
    } else /* eat semicolon */

/** A helper that reduced LLVM versioning in the code. */
inline void ReplaceInstWithValueWrapper(llvm::BasicBlock::iterator &BI, llvm::Value *V) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
    llvm::ReplaceInstWithValue(BI, V);
#else
    llvm::ReplaceInstWithValue(BI->getParent()->getInstList(), BI, V);
#endif
}

} // namespace ispc
