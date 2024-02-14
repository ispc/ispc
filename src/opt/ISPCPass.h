/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
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
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/FunctionAttrs.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#if ISPC_LLVM_VERSION > ISPC_LLVM_17_0
#include <llvm/Transforms/Vectorize/LoadStoreVectorizer.h>
#endif

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
    do {                                                                                                               \
        if (g->debugPrint && (getenv("FUNC") == nullptr ||                                                             \
                              (getenv("FUNC") != nullptr && !strncmp(bb.getParent()->getName().str().c_str(),          \
                                                                     getenv("FUNC"), strlen(getenv("FUNC")))))) {      \
            fprintf(stderr, "Start of " NAME "\n");                                                                    \
            fprintf(stderr, "---------------\n");                                                                      \
            bb.print(llvm::errs());                                                                                    \
            fprintf(stderr, "---------------\n\n");                                                                    \
        }                                                                                                              \
    } while (0)

#define DEBUG_END_BB(NAME)                                                                                             \
    do {                                                                                                               \
        if (g->debugPrint && (getenv("FUNC") == nullptr ||                                                             \
                              (getenv("FUNC") != nullptr && !strncmp(bb.getParent()->getName().str().c_str(),          \
                                                                     getenv("FUNC"), strlen(getenv("FUNC")))))) {      \
            fprintf(stderr, "End of " NAME " %s\n", modifiedAny ? "** CHANGES **" : "");                               \
            fprintf(stderr, "---------------\n");                                                                      \
            bb.print(llvm::errs());                                                                                    \
            fprintf(stderr, "---------------\n\n");                                                                    \
        }                                                                                                              \
    } while (0)

/** A helper that reduced LLVM versioning in the code. */
inline void ReplaceInstWithValueWrapper(llvm::BasicBlock::iterator &BI, llvm::Value *V) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
    llvm::ReplaceInstWithValue(BI, V);
#else
    llvm::ReplaceInstWithValue(BI->getParent()->getInstList(), BI, V);
#endif
}

} // namespace ispc
