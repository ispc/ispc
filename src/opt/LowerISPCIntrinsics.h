/*
  Copyright (c) 2024-2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

struct LowerISPCIntrinsicsPass
#if ISPC_LLVM_VERSION >= ISPC_LLVM_24_0
    : public llvm::RequiredPassInfoMixin<LowerISPCIntrinsicsPass> {
#else
    : public llvm::PassInfoMixin<LowerISPCIntrinsicsPass> {
#endif
    enum class BitcastMode { All, DeferHalfBitcasts, OnlyHalfBitcasts };

    explicit LowerISPCIntrinsicsPass(BitcastMode mode = BitcastMode::All) : m_mode(mode) {}
    static bool isRequired() { return true; }
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    BitcastMode m_mode;
};

} // namespace ispc
