/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

/** ISPC's `inline` qualifier is documented as forcing inlining (not just a
    hint). Historically this was implemented by attaching the LLVM
    `alwaysinline` attribute at parse time, which causes the AlwaysInliner
    sub-pass of every ModuleInlinerWrapperPass invocation to inline the
    function. The first such inliner runs very early in the optimization
    pipeline, before loop optimizations, GVN, jump threading, etc. Inlining a
    large function body into its caller before those optimizations have run on
    the callee tends to produce noticeably worse code: the caller becomes too
    large to optimize well, register pressure climbs, and unnecessary
    broadcasts/blends end up in hot loops (issue #3804).

    Instead, parse-time code attaches a custom string attribute
    `ispc-defer-alwaysinline` and this pass converts it to `alwaysinline` shortly
    before the final inliner pass runs. This delays forced inlining until
    after the callee has been optimized in isolation, while still guaranteeing
    that `inline`-qualified functions are inlined as the language reference
    promises.
 */
class RestoreInlineAttrPass : public llvm::PassInfoMixin<RestoreInlineAttrPass> {
  public:
    RestoreInlineAttrPass() = default;

    static llvm::StringRef getPassName() { return "Restore inline attribute"; }
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

} // namespace ispc
