/*
  Copyright (c) 2010-2022, Intel Corporation
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

/** @file opt.cpp
    @brief Implementations of various ispc optimization passes that operate
           on the LLVM IR.
*/

#include "opt.h"
#include "ctx.h"
#include "llvmutil.h"
#include "module.h"
#include "opt/ISPCPass.h"
#include "opt/ISPCPasses.h"
#include "sym.h"
#include "util.h"

#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdio.h>

#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/IR/Verifier.h>
#include <llvm/InitializePasses.h>
#include <llvm/Pass.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/FunctionAttrs.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/InstSimplifyPass.h>
#if ISPC_LLVM_VERSION >= ISPC_LLVM_15_0
#include <llvm/Transforms/Scalar/SimpleLoopUnswitch.h>
#endif
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Vectorize.h>

#ifdef ISPC_HOST_IS_LINUX
#include <alloca.h>
#elif defined(ISPC_HOST_IS_WINDOWS)
#include <malloc.h>
#ifndef __MINGW32__
#define alloca _alloca
#endif
#endif // ISPC_HOST_IS_WINDOWS

#ifndef PRId64
#define PRId64 "lld"
#endif
#ifndef PRIu64
#define PRIu64 "llu"
#endif

#ifndef ISPC_NO_DUMPS
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Regex.h>
#endif
#ifdef ISPC_XE_ENABLED
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#include <llvm/GenXIntrinsics/GenXIntrOpts.h>
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#include <llvm/GenXIntrinsics/GenXSPIRVWriterAdaptor.h>
// Used for Xe gather coalescing
#include <llvm/Transforms/Utils/Local.h>

#endif

using namespace ispc;

static llvm::Pass *CreatePeepholePass();

static llvm::Pass *CreateGatherCoalescePass();

static llvm::Pass *CreateIsCompileTimeConstantPass(bool isLastTry);
static llvm::Pass *CreateMakeInternalFuncsStaticPass();

#ifndef ISPC_NO_DUMPS
static llvm::Pass *CreateDebugPass(char *output);
static llvm::Pass *CreateDebugPassFile(int number, llvm::StringRef name, std::string dir);
#endif

static llvm::Pass *CreateReplaceStdlibShiftPass();

#ifdef ISPC_XE_ENABLED
static llvm::Pass *CreateXeGatherCoalescingPass();
static llvm::Pass *CreateReplaceLLVMIntrinsics();
static llvm::Pass *CreateDemotePHIs();
static llvm::Pass *CreateCheckIRForGenTarget();
static llvm::Pass *CreateMangleOpenCLBuiltins();
#endif

///////////////////////////////////////////////////////////////////////////
// This is a wrap over class llvm::PassManager. This duplicates PassManager function run()
//   and change PassManager function add by adding some checks and debug passes.
//   This wrap can control:
//   - If we want to switch off optimization with given number.
//   - If we want to dump LLVM IR after optimization with given number.
//   - If we want to generate LLVM IR debug for gdb after optimization with given number.
class DebugPassManager {
  public:
    DebugPassManager() : number(0) {}
    void add(llvm::Pass *P, int stage);
    bool run(llvm::Module &M) { return PM.run(M); }
    llvm::legacy::PassManager &getPM() { return PM; }

  private:
    llvm::legacy::PassManager PM;
    int number;
};

void DebugPassManager::add(llvm::Pass *P, int stage = -1) {
    // taking number of optimization
    if (stage == -1) {
        number++;
    } else {
        number = stage;
    }
    if (g->off_stages.find(number) == g->off_stages.end()) {
        // adding optimization (not switched off)
        PM.add(P);
#ifndef ISPC_NO_DUMPS
        if (g->debug_stages.find(number) != g->debug_stages.end()) {
            // adding dump of LLVM IR after optimization
            if (g->dumpFile) {
                PM.add(CreateDebugPassFile(number, P->getPassName(), g->dumpFilePath));
            } else {
                char buf[100];
                snprintf(buf, sizeof(buf), "\n\n*****LLVM IR after phase %d: %s*****\n\n", number,
                         P->getPassName().data());
                PM.add(CreateDebugPass(buf));
            }
        }
#endif
    }
}
///////////////////////////////////////////////////////////////////////////

void ispc::Optimize(llvm::Module *module, int optLevel) {
#ifndef ISPC_NO_DUMPS
    if (g->debugPrint) {
        printf("*** Code going into optimization ***\n");
        module->dump();
    }
#endif
    DebugPassManager optPM;

    if (g->enableLLVMIntrinsics) {
        // Required for matrix intrinsics. This needs to happen before VerifierPass.
        // TODO : Limit pass to only when llvm.matrix.* intrinsics are used.
        optPM.add(llvm::createLowerMatrixIntrinsicsPass()); // llvm.matrix
    }
    optPM.add(llvm::createVerifierPass(), 0);

    optPM.add(new llvm::TargetLibraryInfoWrapperPass(llvm::Triple(module->getTargetTriple())));
    if (!g->target->isXeTarget()) {
        llvm::TargetMachine *targetMachine = g->target->GetTargetMachine();
        optPM.getPM().add(createTargetTransformInfoWrapperPass(targetMachine->getTargetIRAnalysis()));
    }
    optPM.add(llvm::createIndVarSimplifyPass());

#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
    llvm::SimplifyCFGOptions simplifyCFGopt;
    simplifyCFGopt.HoistCommonInsts = true;
#endif
    if (optLevel == 0) {
        // This is more or less the minimum set of optimizations that we
        // need to do to generate code that will actually run.  (We can't
        // run absolutely no optimizations, since the front-end needs us to
        // take the various __pseudo_* functions it has emitted and turn
        // them into something that can actually execute.
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            // Global DCE is required for ISPCSimdCFLoweringPass
            optPM.add(llvm::createGlobalDCEPass());
            // FIXME: temporary solution
            optPM.add(llvm::createBreakCriticalEdgesPass());
            optPM.add(CreateDemotePHIs());
            optPM.add(llvm::createISPCSimdCFLoweringPass());
            // FIXME: temporary solution
            optPM.add(llvm::createPromoteMemoryToRegisterPass());
        }
#endif
        optPM.add(CreateImproveMemoryOpsPass(), 100);

        if (g->opt.disableHandlePseudoMemoryOps == false)
            optPM.add(CreateReplacePseudoMemoryOpsPass());

        optPM.add(CreateIntrinsicsOptPass(), 102);
        optPM.add(CreateIsCompileTimeConstantPass(true));
        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(CreateMakeInternalFuncsStaticPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.add(llvm::createPromoteMemoryToRegisterPass());
            // This pass is needed for correct prints work
            optPM.add(llvm::createSROAPass());
            optPM.add(CreateReplaceLLVMIntrinsics());
            optPM.add(CreateCheckIRForGenTarget());
            optPM.add(CreateMangleOpenCLBuiltins());
            // This pass is required to prepare LLVM IR for open source SPIR-V translator
            optPM.add(
                llvm::createGenXSPIRVWriterAdaptorPass(true /*RewriteTypes*/, false /*RewriteSingleElementVectors*/));
        }
#endif
        optPM.add(llvm::createGlobalDCEPass());
    } else {
        llvm::PassRegistry *registry = llvm::PassRegistry::getPassRegistry();
        llvm::initializeCore(*registry);
        llvm::initializeScalarOpts(*registry);
        llvm::initializeIPO(*registry);
        llvm::initializeAnalysis(*registry);
        llvm::initializeTransformUtils(*registry);
        llvm::initializeInstCombine(*registry);
        llvm::initializeInstrumentation(*registry);
        llvm::initializeTarget(*registry);

        optPM.add(llvm::createGlobalDCEPass(), 184);

#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            // FIXME: temporary solution
            optPM.add(llvm::createBreakCriticalEdgesPass());
            optPM.add(CreateDemotePHIs());
            optPM.add(llvm::createISPCSimdCFLoweringPass());
            // FIXME: temporary solution
            optPM.add(llvm::createPromoteMemoryToRegisterPass());
        }
#endif
        // Setup to use LLVM default AliasAnalysis
        // Ideally, we want call:
        //    llvm::PassManagerBuilder pm_Builder;
        //    pm_Builder.OptLevel = optLevel;
        //    pm_Builder.addInitialAliasAnalysisPasses(optPM);
        // but the addInitialAliasAnalysisPasses() is a private function
        // so we explicitly enable them here.
        // Need to keep sync with future LLVM change
        // An alternative is to call populateFunctionPassManager()
        optPM.add(llvm::createTypeBasedAAWrapperPass(), 190);
        optPM.add(llvm::createBasicAAWrapperPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif

        optPM.add(llvm::createSROAPass());

        optPM.add(llvm::createEarlyCSEPass());
        optPM.add(llvm::createLowerExpectIntrinsicPass());

        // Early optimizations to try to reduce the total amount of code to
        // work with if we can
        optPM.add(llvm::createReassociatePass(), 200);
        optPM.add(llvm::createInstSimplifyLegacyPass());
        optPM.add(llvm::createDeadCodeEliminationPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif

        optPM.add(llvm::createPromoteMemoryToRegisterPass());
        optPM.add(llvm::createAggressiveDCEPass());

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.add(llvm::createInstructionCombiningPass(), 210);
            optPM.add(CreateImproveMemoryOpsPass());
        }
        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.add(CreateIntrinsicsOptPass(), 215);
            optPM.add(CreateInstructionSimplifyPass());
        }
        optPM.add(llvm::createDeadCodeEliminationPass(), 220);

        // On to more serious optimizations
        optPM.add(llvm::createSROAPass());
        optPM.add(llvm::createInstructionCombiningPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif
        optPM.add(llvm::createPromoteMemoryToRegisterPass());
        optPM.add(llvm::createGlobalOptimizerPass());
        optPM.add(llvm::createReassociatePass());
        // IPConstProp will not be supported by LLVM moving forward.
        // Switching to IPSCCP which is its recommended functional equivalent.
        // TODO : Make IPSCCP the default after ISPC 1.14 release.
#if ISPC_LLVM_VERSION < ISPC_LLVM_12_0
        optPM.add(llvm::createIPConstantPropagationPass());
#else
        optPM.add(llvm::createIPSCCPPass());
#endif

        optPM.add(CreateReplaceStdlibShiftPass(), 229);

        optPM.add(llvm::createDeadArgEliminationPass(), 230);
        optPM.add(llvm::createInstructionCombiningPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif
        optPM.add(llvm::createPruneEHPass());
        optPM.add(llvm::createPostOrderFunctionAttrsLegacyPass());
        optPM.add(llvm::createReversePostOrderFunctionAttrsPass());

        // Next inline pass will remove functions, saved by __keep_funcs_live
        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(llvm::createInstSimplifyLegacyPass());
        optPM.add(llvm::createDeadCodeEliminationPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif

#if ISPC_LLVM_VERSION < ISPC_LLVM_15_0
        // Starting LLVM 15.0 this pass is supported with new pass manager only (217e857)
        // TODO: switch ISPC to new pass manager: https://github.com/ispc/ispc/issues/2359
        optPM.add(llvm::createArgumentPromotionPass());
#endif

        optPM.add(llvm::createAggressiveDCEPass());
        optPM.add(llvm::createInstructionCombiningPass(), 241);
        optPM.add(llvm::createJumpThreadingPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif

        optPM.add(llvm::createSROAPass());

        optPM.add(llvm::createInstructionCombiningPass());
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            // Inline
            optPM.add(llvm::createCorrelatedValuePropagationPass());
            optPM.add(llvm::createInstructionCombiningPass());
            optPM.add(llvm::createGlobalDCEPass());
            optPM.add(llvm::createInstructionCombiningPass());
            optPM.add(llvm::createEarlyCSEPass());
            optPM.add(llvm::createDeadCodeEliminationPass());
        }
#endif
        optPM.add(llvm::createTailCallEliminationPass());

        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.add(CreateIntrinsicsOptPass(), 250);
            optPM.add(CreateInstructionSimplifyPass());
        }

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.add(llvm::createInstructionCombiningPass(), 255);
            optPM.add(CreateImproveMemoryOpsPass());

            if (g->opt.disableCoalescing == false) {
                // It is important to run this here to make it easier to
                // finding matching gathers we can coalesce..
                optPM.add(llvm::createEarlyCSEPass(), 260);
                optPM.add(CreateGatherCoalescePass());
            }
        }

        optPM.add(llvm::createFunctionInliningPass(), 265);
        optPM.add(llvm::createInstSimplifyLegacyPass());
        optPM.add(CreateIntrinsicsOptPass());
        optPM.add(CreateInstructionSimplifyPass());

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.add(llvm::createInstructionCombiningPass(), 270);
            optPM.add(CreateImproveMemoryOpsPass());
        }

        optPM.add(llvm::createIPSCCPPass(), 275);
        optPM.add(llvm::createDeadArgEliminationPass());
        optPM.add(llvm::createAggressiveDCEPass());
        optPM.add(llvm::createInstructionCombiningPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif

        if (g->opt.disableHandlePseudoMemoryOps == false) {
            optPM.add(CreateReplacePseudoMemoryOpsPass(), 280);
        }
        optPM.add(CreateIntrinsicsOptPass(), 281);
        optPM.add(CreateInstructionSimplifyPass());

        optPM.add(llvm::createFunctionInliningPass());
#if ISPC_LLVM_VERSION < ISPC_LLVM_15_0
        // Starting LLVM 15.0 this pass is supported with new pass manager only (217e857)
        // TODO: switch ISPC to new pass manager: https://github.com/ispc/ispc/issues/2359
        optPM.add(llvm::createArgumentPromotionPass());
#endif

        optPM.add(llvm::createSROAPass());

        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(CreateInstructionSimplifyPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif
        optPM.add(llvm::createReassociatePass());
        optPM.add(llvm::createLoopRotatePass());
        optPM.add(llvm::createLICMPass());
        // Loop unswitch pass was removed in LLVM 15.0 (fb4113).
        // Recommended replacement: createSimpleLoopUnswitchLegacyPass
#if ISPC_LLVM_VERSION < ISPC_LLVM_15_0
        optPM.add(llvm::createLoopUnswitchPass(false));
#else
        optPM.add(llvm::createSimpleLoopUnswitchLegacyPass(false));
#endif
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(CreateInstructionSimplifyPass());
        optPM.add(llvm::createIndVarSimplifyPass());
        // Currently CM does not support memset/memcpy
        // so this pass is temporary disabled for Xe.
        if (!g->target->isXeTarget()) {
            optPM.add(llvm::createLoopIdiomPass());
        }
        optPM.add(llvm::createLoopDeletionPass());
        if (g->opt.unrollLoops) {
            optPM.add(llvm::createLoopUnrollPass(), 300);
        }
        optPM.add(llvm::createGVNPass(), 301);

        optPM.add(CreateIsCompileTimeConstantPass(true));
        optPM.add(CreateIntrinsicsOptPass());
        optPM.add(CreateInstructionSimplifyPass());

#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget() && g->opt.disableGatherScatterOptimizations == false &&
            g->target->getVectorWidth() > 1) {
            if (!g->opt.disableXeGatherCoalescing) {
                optPM.add(CreateXeGatherCoalescingPass(), 321);

                // Try the llvm provided load/store vectorizer
                optPM.add(llvm::createLoadStoreVectorizerPass(), 325);
            }
        }
#endif

        // Currently CM does not support memset/memcpy
        // so this pass is temporary disabled for Xe.
        if (!g->target->isXeTarget()) {
            optPM.add(llvm::createMemCpyOptPass());
        }
        optPM.add(llvm::createSCCPPass());
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(CreateInstructionSimplifyPass());
        optPM.add(llvm::createJumpThreadingPass());
        optPM.add(llvm::createCorrelatedValuePropagationPass());
        optPM.add(llvm::createDeadStoreEliminationPass());
        optPM.add(llvm::createAggressiveDCEPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(CreateInstructionSimplifyPass());
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.add(CreateReplaceLLVMIntrinsics());
        }
#endif
        optPM.add(CreatePeepholePass());
        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(llvm::createAggressiveDCEPass());
        optPM.add(llvm::createStripDeadPrototypesPass());
        optPM.add(CreateMakeInternalFuncsStaticPass());
        optPM.add(llvm::createGlobalDCEPass());
        optPM.add(llvm::createConstantMergePass());
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.add(CreateCheckIRForGenTarget());
            optPM.add(CreateMangleOpenCLBuiltins());
            // This pass is required to prepare LLVM IR for open source SPIR-V translator
            optPM.add(
                llvm::createGenXSPIRVWriterAdaptorPass(true /*RewriteTypes*/, false /*RewriteSingleElementVectors*/));
        }
#endif
    }

    // Finish up by making sure we didn't mess anything up in the IR along
    // the way.
    optPM.add(llvm::createVerifierPass(), LAST_OPT_NUMBER);
    optPM.run(*module);

#ifndef ISPC_NO_DUMPS
    if (g->debugPrint) {
        printf("\n*****\nFINAL OUTPUT\n*****\n");
        module->dump();
    }
#endif
}

///////////////////////////////////////////////////////////////////////////
// GatherCoalescePass

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
    GatherCoalescePass() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const { return "Gather Coalescing"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);

  private:
    // Type of base pointer element type (the 1st argument of the intrinsic) is i8
    // e.g. @__pseudo_gather_factored_base_offsets32_i32(i8 *, <WIDTH x i32>, i32, <WIDTH x i32>, <WIDTH x MASK>)
    llvm::Type *baseType{LLVMTypes::Int8Type};
};

char GatherCoalescePass::ID = 0;

/** Representation of a memory load that the gather coalescing code has
    decided to generate.
 */
struct CoalescedLoadOp {
    CoalescedLoadOp(int64_t s, int c) {
        start = s;
        count = c;
        load = element0 = element1 = NULL;
    }

    /** Starting offset of the load from the common base pointer (in terms
        of numbers of items of the underlying element type--*not* in terms
        of bytes). */
    int64_t start;

    /** Number of elements to load at this location */
    int count;

    /** Value loaded from memory for this load op */
    llvm::Value *load;

    /** For 2-wide loads (i.e. 64-bit loads), these store the lower and
        upper 32 bits of the result, respectively. */
    llvm::Value *element0, *element1;
};

/** This function determines whether it makes sense (and is safe) to
    generate a vector load of width vectorWidth, starting at *iter.  It
    returns true if so, setting *newIter to point to the next element in
    the set that isn't taken care of by the generated load.  If a vector
    load of the given width doesn't make sense, then false is returned.
 */
static bool lVectorLoadIsEfficient(std::set<int64_t>::iterator iter, std::set<int64_t>::iterator end,
                                   std::set<int64_t>::iterator *newIter, int vectorWidth) {
    // We're considering a vector load of width vectorWidth, starting at
    // the offset "start".
    int64_t start = *iter;

    // The basic idea is that we'll look at the subsequent elements in the
    // load set after the initial one at start.  As long as subsequent
    // elements:
    //
    // 1. Aren't so far separated that they no longer fit into the range
    //    [start, start+vectorWidth)
    //
    // 2. And don't have too large a gap in between them (e.g., it's not
    //    worth generating an 8-wide load for two elements with offsets 0
    //    and 7, but no loads requested in between).
    //
    // Then we continue moving forward through the elements until we either
    // fill up the vector or run out of elements.

    // lastAccepted holds the last offset we've processed and accepted as
    // valid for the vector load underconsideration
    int64_t lastAccepted = start;

    while (iter != end) {
        // What is the separation in offset values from the last element we
        // added to the set for this load?
        int64_t delta = *iter - lastAccepted;
        if (delta > 3)
            // If there's too big a gap, then we won't issue the load
            return false;

        int64_t span = *iter - start + 1;

        if (span == vectorWidth) {
            // We've extended far enough that we have exactly filled up the
            // entire vector width; we can't go any further, so return with
            // success.  (Update *newIter to point at the next element
            // after the last one accepted here.)
            *newIter = ++iter;
            return true;
        } else if (span > vectorWidth) {
            // The current offset won't fit into a vectorWidth-wide load
            // starting from start.  It's still generally worthwhile
            // issuing the load we've been considering, though, since it
            // will provide values for a number of previous offsets.  This
            // load will have one or more elements at the end of its range
            // that is not needed by any of the offsets under
            // consideration.  As such, there are three cases where issuing
            // this load is a bad idea:
            //
            // 1. 2-wide loads: we know that we haven't completely filled
            //    the 2-wide vector, since otherwise the if() test above
            //    would have succeeded previously.  Therefore, we must have
            //    a situation with offsets like (4,6,...); it would be a
            //    silly idea to issue a 2-wide load to get the value for
            //    the 4 offset, versus failing here and issuing a scalar
            //    load instead.
            //
            // 2. If there are too many unnecessary values at the end of
            //    the load extent (defined as more than half of them)--in
            //    this case, it'd be better to issue a vector load of
            //    smaller width anyway.
            //
            // 3. If the gap between the last accepted offset and the
            //    current one under consideration is more than the page
            //    size.  In this case we can't be sure whether or not some
            //    of the unused elements at the end of the load will
            //    straddle a page boundary and thus lead to an undesirable
            //    fault.  (It's hard to imagine this happening in practice,
            //    except under contrived circumstances, but better safe
            //    than sorry.)
            const int pageSize = 4096;
            if (vectorWidth != 2 && (lastAccepted - start) > (vectorWidth / 2) && (*iter - lastAccepted) < pageSize) {
                *newIter = iter;
                return true;
            } else
                return false;
        }

        // Continue moving forward
        lastAccepted = *iter;
        ++iter;
    }

    return false;
}

/** Given a set of offsets from a common base pointer that we need to get
    loaded into memory, determine a reasonable set of load operations that
    gets all of the corresponding values in memory (ideally, including as
    many as possible wider vector loads rather than scalar loads).  Return
    a CoalescedLoadOp for each one in the *loads array.
 */
static void lSelectLoads(const std::vector<int64_t> &loadOffsets, std::vector<CoalescedLoadOp> *loads) {
    // First, get a sorted set of unique offsets to load from.
    std::set<int64_t> allOffsets;
    for (unsigned int i = 0; i < loadOffsets.size(); ++i)
        allOffsets.insert(loadOffsets[i]);

    std::set<int64_t>::iterator iter = allOffsets.begin();
    while (iter != allOffsets.end()) {
        Debug(SourcePos(), "Load needed at %" PRId64 ".", *iter);
        ++iter;
    }

    // Now, iterate over the offsets from low to high.  Starting at the
    // current offset, we see if a vector load starting from that offset
    // will cover loads at subsequent offsets as well.
    iter = allOffsets.begin();
    while (iter != allOffsets.end()) {
        // Consider vector loads of width of each of the elements of
        // spanSizes[], in order.
        int vectorWidths[] = {8, 4, 2};
        int nVectorWidths = sizeof(vectorWidths) / sizeof(vectorWidths[0]);
        bool gotOne = false;
        for (int i = 0; i < nVectorWidths; ++i) {
            // See if a load of vector with width vectorWidths[i] would be
            // effective (i.e. would cover a reasonable number of the
            // offsets that need to be loaded from).
            std::set<int64_t>::iterator newIter;
            if (lVectorLoadIsEfficient(iter, allOffsets.end(), &newIter, vectorWidths[i])) {
                // Yes: create the corresponding coalesced load and update
                // the iterator to the returned iterator; doing so skips
                // over the additional offsets that are taken care of by
                // this load.
                loads->push_back(CoalescedLoadOp(*iter, vectorWidths[i]));
                iter = newIter;
                gotOne = true;
                break;
            }
        }

        if (gotOne == false) {
            // We couldn't find a vector load starting from this offset
            // that made sense, so emit a scalar load and continue onward.
            loads->push_back(CoalescedLoadOp(*iter, 1));
            ++iter;
        }
    }
}

/** Print a performance message with the details of the result of
    coalescing over a group of gathers. */
static void lCoalescePerfInfo(const std::vector<llvm::CallInst *> &coalesceGroup,
                              const std::vector<CoalescedLoadOp> &loadOps) {
    SourcePos pos;
    LLVMGetSourcePosFromMetadata(coalesceGroup[0], &pos);

    // Create a string that indicates the line numbers of the subsequent
    // gathers from the first one that were coalesced here.
    char otherPositions[512];
    otherPositions[0] = '\0';
    if (coalesceGroup.size() > 1) {
        const char *plural = (coalesceGroup.size() > 2) ? "s" : "";
        char otherBuf[32];
        snprintf(otherBuf, sizeof(otherBuf), "(other%s at line%s ", plural, plural);
        strncat(otherPositions, otherBuf, sizeof(otherPositions) - strlen(otherPositions) - 1);

        for (int i = 1; i < (int)coalesceGroup.size(); ++i) {
            SourcePos p;
            bool ok = LLVMGetSourcePosFromMetadata(coalesceGroup[i], &p);
            if (ok) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%d", p.first_line);
                strncat(otherPositions, buf, sizeof(otherPositions) - strlen(otherPositions) - 1);
                if (i < (int)coalesceGroup.size() - 1)
                    strncat(otherPositions, ", ", sizeof(otherPositions) - strlen(otherPositions) - 1);
            }
        }
        strncat(otherPositions, ") ", sizeof(otherPositions) - strlen(otherPositions) - 1);
    }

    // Count how many loads of each size there were.
    std::map<int, int> loadOpsCount;
    for (int i = 0; i < (int)loadOps.size(); ++i)
        ++loadOpsCount[loadOps[i].count];

    // Generate a string the describes the mix of load ops
    char loadOpsInfo[512];
    loadOpsInfo[0] = '\0';
    std::map<int, int>::const_iterator iter = loadOpsCount.begin();
    while (iter != loadOpsCount.end()) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d x %d-wide", iter->second, iter->first);
        if ((strlen(loadOpsInfo) + strlen(buf)) >= 512) {
            break;
        }
        strncat(loadOpsInfo, buf, sizeof(loadOpsInfo) - strlen(loadOpsInfo) - 1);
        ++iter;
        if (iter != loadOpsCount.end())
            strncat(loadOpsInfo, ", ", sizeof(loadOpsInfo) - strlen(loadOpsInfo) - 1);
    }

    if (g->opt.level > 0) {
        if (coalesceGroup.size() == 1)
            PerformanceWarning(pos, "Coalesced gather into %d load%s (%s).", (int)loadOps.size(),
                               (loadOps.size() > 1) ? "s" : "", loadOpsInfo);
        else
            PerformanceWarning(pos,
                               "Coalesced %d gathers starting here %sinto %d "
                               "load%s (%s).",
                               (int)coalesceGroup.size(), otherPositions, (int)loadOps.size(),
                               (loadOps.size() > 1) ? "s" : "", loadOpsInfo);
    }
}

/** Utility routine that computes an offset from a base pointer and then
    returns the result of a load of the given type from the resulting
    location:

    return *((type *)(basePtr + offset))
 */
llvm::Value *lGEPAndLoad(llvm::Value *basePtr, llvm::Type *baseType, int64_t offset, int align,
                         llvm::Instruction *insertBefore, llvm::Type *type) {
    llvm::Value *ptr = LLVMGEPInst(basePtr, baseType, LLVMInt64(offset), "new_base", insertBefore);
    ptr = new llvm::BitCastInst(ptr, llvm::PointerType::get(type, 0), "ptr_cast", insertBefore);
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
    return new llvm::LoadInst(ptr, "gather_load", false /* not volatile */, llvm::MaybeAlign(align), insertBefore);
#else // LLVM 11.0+
    Assert(llvm::isa<llvm::PointerType>(ptr->getType()));
    return new llvm::LoadInst(type, ptr, "gather_load", false /* not volatile */, llvm::MaybeAlign(align).valueOrOne(),
                              insertBefore);
#endif
}

/* Having decided that we're doing to emit a series of loads, as encoded in
   the loadOps array, this function emits the corresponding load
   instructions.
 */
static void lEmitLoads(llvm::Value *basePtr, llvm::Type *baseType, std::vector<CoalescedLoadOp> &loadOps,
                       int elementSize, llvm::Instruction *insertBefore) {
    Debug(SourcePos(), "Coalesce doing %d loads.", (int)loadOps.size());
    for (int i = 0; i < (int)loadOps.size(); ++i) {
        Debug(SourcePos(), "Load #%d @ %" PRId64 ", %d items", i, loadOps[i].start, loadOps[i].count);

        // basePtr is an i8 *, so the offset from it should be in terms of
        // bytes, not underlying i32 elements.
        int64_t start = loadOps[i].start * elementSize;

        int align = 4;
        switch (loadOps[i].count) {
        case 1:
            // Single 32-bit scalar load
            loadOps[i].load = lGEPAndLoad(basePtr, baseType, start, align, insertBefore, LLVMTypes::Int32Type);
            break;
        case 2: {
            // Emit 2 x i32 loads as i64 loads and then break the result
            // into two 32-bit parts.
            loadOps[i].load = lGEPAndLoad(basePtr, baseType, start, align, insertBefore, LLVMTypes::Int64Type);
            // element0 = (int32)value;
            loadOps[i].element0 =
                new llvm::TruncInst(loadOps[i].load, LLVMTypes::Int32Type, "load64_elt0", insertBefore);
            // element1 = (int32)(value >> 32)
            llvm::Value *shift = llvm::BinaryOperator::Create(llvm::Instruction::LShr, loadOps[i].load, LLVMInt64(32),
                                                              "load64_shift", insertBefore);
            loadOps[i].element1 = new llvm::TruncInst(shift, LLVMTypes::Int32Type, "load64_elt1", insertBefore);
            break;
        }
        case 4: {
            // 4-wide vector load
            if (g->opt.forceAlignedMemory) {
                align = g->target->getNativeVectorAlignment();
            }
            llvm::VectorType *vt = LLVMVECTOR::get(LLVMTypes::Int32Type, 4);
            loadOps[i].load = lGEPAndLoad(basePtr, baseType, start, align, insertBefore, vt);
            break;
        }
        case 8: {
            // 8-wide vector load
            if (g->opt.forceAlignedMemory) {
                align = g->target->getNativeVectorAlignment();
            }
            llvm::VectorType *vt = LLVMVECTOR::get(LLVMTypes::Int32Type, 8);
            loadOps[i].load = lGEPAndLoad(basePtr, baseType, start, align, insertBefore, vt);
            break;
        }
        default:
            FATAL("Unexpected load count in lEmitLoads()");
        }
    }
}

/** Convert any loads of 8-wide vectors into two 4-wide vectors
    (logically).  This allows the assembly code below to always operate on
    4-wide vectors, which leads to better code.  Returns a new vector of
    load operations.
 */
static std::vector<CoalescedLoadOp> lSplit8WideLoads(const std::vector<CoalescedLoadOp> &loadOps,
                                                     llvm::Instruction *insertBefore) {
    std::vector<CoalescedLoadOp> ret;
    for (unsigned int i = 0; i < loadOps.size(); ++i) {
        if (loadOps[i].count == 8) {
            // Create fake CoalescedLOadOps, where the load llvm::Value is
            // actually a shuffle that pulls either the first 4 or the last
            // 4 values out of the original 8-wide loaded value.
            int32_t shuf[2][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}};

            ret.push_back(CoalescedLoadOp(loadOps[i].start, 4));
            ret.back().load = LLVMShuffleVectors(loadOps[i].load, loadOps[i].load, shuf[0], 4, insertBefore);

            ret.push_back(CoalescedLoadOp(loadOps[i].start + 4, 4));
            ret.back().load = LLVMShuffleVectors(loadOps[i].load, loadOps[i].load, shuf[1], 4, insertBefore);
        } else
            ret.push_back(loadOps[i]);
    }

    return ret;
}

/** Given a 1-wide load of a 32-bit value, merge its value into the result
    vector for any and all elements for which it applies.
 */
static llvm::Value *lApplyLoad1(llvm::Value *result, const CoalescedLoadOp &load, const int64_t offsets[4], bool set[4],
                                llvm::Instruction *insertBefore) {
    for (int elt = 0; elt < 4; ++elt) {
        if (offsets[elt] >= load.start && offsets[elt] < load.start + load.count) {
            Debug(SourcePos(),
                  "Load 1 @ %" PRId64 " matches for element #%d "
                  "(value %" PRId64 ")",
                  load.start, elt, offsets[elt]);
            // If this load gives one of the values that we need, then we
            // can just insert it in directly
            Assert(set[elt] == false);
            result = llvm::InsertElementInst::Create(result, load.load, LLVMInt32(elt), "insert_load", insertBefore);
            set[elt] = true;
        }
    }

    return result;
}

/** Similarly, incorporate the values from a 2-wide load into any vector
    elements that they apply to. */
static llvm::Value *lApplyLoad2(llvm::Value *result, const CoalescedLoadOp &load, const int64_t offsets[4], bool set[4],
                                llvm::Instruction *insertBefore) {
    int elt = 0;
    while (elt < 4) {
        // First, try to do a 64-bit-wide insert into the result vector.
        // We can do this when we're currently at an even element, when the
        // current and next element have consecutive values, and where the
        // original 64-bit load is at the offset needed by the current
        // element.
        if ((elt & 1) == 0 && offsets[elt] + 1 == offsets[elt + 1] && offsets[elt] == load.start) {
            Debug(SourcePos(),
                  "Load 2 @ %" PRId64 " matches for elements #%d,%d "
                  "(values %" PRId64 ",%" PRId64 ")",
                  load.start, elt, elt + 1, offsets[elt], offsets[elt + 1]);
            Assert(set[elt] == false && ((elt < 3) && set[elt + 1] == false));

            // In this case, we bitcast from a 4xi32 to a 2xi64 vector
            llvm::Type *vec2x64Type = LLVMVECTOR::get(LLVMTypes::Int64Type, 2);
            result = new llvm::BitCastInst(result, vec2x64Type, "to2x64", insertBefore);

            // And now we can insert the 64-bit wide value into the
            // appropriate elment
            result = llvm::InsertElementInst::Create(result, load.load, LLVMInt32(elt / 2), "insert64", insertBefore);

            // And back to 4xi32.
            llvm::Type *vec4x32Type = LLVMVECTOR::get(LLVMTypes::Int32Type, 4);
            result = new llvm::BitCastInst(result, vec4x32Type, "to4x32", insertBefore);

            set[elt] = true;
            if (elt < 3) {
                set[elt + 1] = true;
            }
            // Advance elt one extra time, since we just took care of two
            // elements
            ++elt;
        } else if (offsets[elt] >= load.start && offsets[elt] < load.start + load.count) {
            Debug(SourcePos(),
                  "Load 2 @ %" PRId64 " matches for element #%d "
                  "(value %" PRId64 ")",
                  load.start, elt, offsets[elt]);
            // Otherwise, insert one of the 32-bit pieces into an element
            // of the final vector
            Assert(set[elt] == false);
            llvm::Value *toInsert = (offsets[elt] == load.start) ? load.element0 : load.element1;
            result = llvm::InsertElementInst::Create(result, toInsert, LLVMInt32(elt), "insert_load", insertBefore);
            set[elt] = true;
        }
        ++elt;
    }

    return result;
}

#if 1
/* This approach works better with AVX, while the #else path generates
   slightly better code with SSE.  Need to continue to dig into performance
   details with this stuff in general... */

/** And handle a 4-wide load */
static llvm::Value *lApplyLoad4(llvm::Value *result, const CoalescedLoadOp &load, const int64_t offsets[4], bool set[4],
                                llvm::Instruction *insertBefore) {
    // Conceptually, we're doing to consider doing a shuffle vector with
    // the 4-wide load and the 4-wide result we have so far to generate a
    // new 4-wide vector.  We'll start with shuffle indices that just
    // select each element of the result so far for the result.
    int32_t shuf[4] = {4, 5, 6, 7};

    for (int elt = 0; elt < 4; ++elt) {
        if (offsets[elt] >= load.start && offsets[elt] < load.start + load.count) {
            Debug(SourcePos(),
                  "Load 4 @ %" PRId64 " matches for element #%d "
                  "(value %" PRId64 ")",
                  load.start, elt, offsets[elt]);

            // If the current element falls within the range of locations
            // that the 4-wide load covers, then compute the appropriate
            // shuffle index that extracts the appropriate element from the
            // load.
            Assert(set[elt] == false);
            shuf[elt] = int32_t(offsets[elt] - load.start);
            set[elt] = true;
        }
    }

    // Now, issue a shufflevector instruction if any of the values from the
    // load we just considered were applicable.
    if (shuf[0] != 4 || shuf[1] != 5 || shuf[2] != 6 || shuf[3] != 7)
        result = LLVMShuffleVectors(load.load, result, shuf, 4, insertBefore);

    return result;
}

/** We're need to fill in the values for a 4-wide result vector.  This
    function looks at all of the generated loads and extracts the
    appropriate elements from the appropriate loads to assemble the result.
    Here the offsets[] parameter gives the 4 offsets from the base pointer
    for the four elements of the result.
*/
static llvm::Value *lAssemble4Vector(const std::vector<CoalescedLoadOp> &loadOps, const int64_t offsets[4],
                                     llvm::Instruction *insertBefore) {
    llvm::Type *returnType = LLVMVECTOR::get(LLVMTypes::Int32Type, 4);
    llvm::Value *result = llvm::UndefValue::get(returnType);

    Debug(SourcePos(), "Starting search for loads [%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "].", offsets[0],
          offsets[1], offsets[2], offsets[3]);

    // Track whether we have found a valid value for each of the four
    // elements of the result
    bool set[4] = {false, false, false, false};

    // Loop over all of the loads and check each one to see if it provides
    // a value that's applicable to the result
    for (int load = 0; load < (int)loadOps.size(); ++load) {
        const CoalescedLoadOp &li = loadOps[load];

        switch (li.count) {
        case 1:
            result = lApplyLoad1(result, li, offsets, set, insertBefore);
            break;
        case 2:
            result = lApplyLoad2(result, li, offsets, set, insertBefore);
            break;
        case 4:
            result = lApplyLoad4(result, li, offsets, set, insertBefore);
            break;
        default:
            FATAL("Unexpected load count in lAssemble4Vector()");
        }
    }

    Debug(SourcePos(), "Done with search for loads [%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "].", offsets[0],
          offsets[1], offsets[2], offsets[3]);

    for (int i = 0; i < 4; ++i)
        Assert(set[i] == true);

    return result;
}

#else

static llvm::Value *lApplyLoad4s(llvm::Value *result, const std::vector<CoalescedLoadOp> &loadOps,
                                 const int64_t offsets[4], bool set[4], llvm::Instruction *insertBefore) {
    int32_t firstMatchElements[4] = {-1, -1, -1, -1};
    const CoalescedLoadOp *firstMatch = NULL;

    Assert(llvm::isa<llvm::UndefValue>(result));

    for (int load = 0; load < (int)loadOps.size(); ++load) {
        const CoalescedLoadOp &loadop = loadOps[load];
        if (loadop.count != 4)
            continue;

        int32_t matchElements[4] = {-1, -1, -1, -1};
        bool anyMatched = false;
        for (int elt = 0; elt < 4; ++elt) {
            if (offsets[elt] >= loadop.start && offsets[elt] < loadop.start + loadop.count) {
                Debug(SourcePos(),
                      "Load 4 @ %" PRId64 " matches for element #%d "
                      "(value %" PRId64 ")",
                      loadop.start, elt, offsets[elt]);
                anyMatched = true;
                Assert(set[elt] == false);
                matchElements[elt] = offsets[elt] - loadop.start;
                set[elt] = true;
            }
        }

        if (anyMatched) {
            if (llvm::isa<llvm::UndefValue>(result)) {
                if (firstMatch == NULL) {
                    firstMatch = &loadop;
                    for (int i = 0; i < 4; ++i)
                        firstMatchElements[i] = matchElements[i];
                } else {
                    int32_t shuffle[4] = {-1, -1, -1, -1};
                    for (int i = 0; i < 4; ++i) {
                        if (firstMatchElements[i] != -1)
                            shuffle[i] = firstMatchElements[i];
                        else
                            shuffle[i] = 4 + matchElements[i];
                    }
                    result = LLVMShuffleVectors(firstMatch->load, loadop.load, shuffle, 4, insertBefore);
                    firstMatch = NULL;
                }
            } else {
                int32_t shuffle[4] = {-1, -1, -1, -1};
                for (int i = 0; i < 4; ++i) {
                    if (matchElements[i] != -1)
                        shuffle[i] = 4 + matchElements[i];
                    else
                        shuffle[i] = i;
                }
                result = LLVMShuffleVectors(result, loadop.load, shuffle, 4, insertBefore);
            }
        }
    }

    if (firstMatch != NULL && llvm::isa<llvm::UndefValue>(result))
        return LLVMShuffleVectors(firstMatch->load, result, firstMatchElements, 4, insertBefore);
    else
        return result;
}

static llvm::Value *lApplyLoad12s(llvm::Value *result, const std::vector<CoalescedLoadOp> &loadOps,
                                  const int64_t offsets[4], bool set[4], llvm::Instruction *insertBefore) {
    // Loop over all of the loads and check each one to see if it provides
    // a value that's applicable to the result
    for (int load = 0; load < (int)loadOps.size(); ++load) {
        const CoalescedLoadOp &loadop = loadOps[load];
        Assert(loadop.count == 1 || loadop.count == 2 || loadop.count == 4);

        if (loadop.count == 1)
            result = lApplyLoad1(result, loadop, offsets, set, insertBefore);
        else if (loadop.count == 2)
            result = lApplyLoad2(result, loadop, offsets, set, insertBefore);
    }
    return result;
}

/** We're need to fill in the values for a 4-wide result vector.  This
    function looks at all of the generated loads and extracts the
    appropriate elements from the appropriate loads to assemble the result.
    Here the offsets[] parameter gives the 4 offsets from the base pointer
    for the four elements of the result.
*/
static llvm::Value *lAssemble4Vector(const std::vector<CoalescedLoadOp> &loadOps, const int64_t offsets[4],
                                     llvm::Instruction *insertBefore) {
    llvm::Type *returnType = LLVMVECTOR::get(LLVMTypes::Int32Type, 4);
    llvm::Value *result = llvm::UndefValue::get(returnType);

    Debug(SourcePos(), "Starting search for loads [%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "].", offsets[0],
          offsets[1], offsets[2], offsets[3]);

    // Track whether we have found a valid value for each of the four
    // elements of the result
    bool set[4] = {false, false, false, false};

    result = lApplyLoad4s(result, loadOps, offsets, set, insertBefore);
    result = lApplyLoad12s(result, loadOps, offsets, set, insertBefore);

    Debug(SourcePos(), "Done with search for loads [%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "].", offsets[0],
          offsets[1], offsets[2], offsets[3]);

    for (int i = 0; i < 4; ++i)
        Assert(set[i] == true);

    return result;
}
#endif

/** Given the set of loads that we've done and the set of result values to
    be computed, this function computes the final llvm::Value *s for each
    result vector.
 */
static void lAssembleResultVectors(const std::vector<CoalescedLoadOp> &loadOps,
                                   const std::vector<int64_t> &constOffsets, std::vector<llvm::Value *> &results,
                                   llvm::Instruction *insertBefore) {
    // We work on 4-wide chunks of the final values, even when we're
    // computing 8-wide or 16-wide vectors.  This gives better code from
    // LLVM's SSE/AVX code generators.
    Assert((constOffsets.size() % 4) == 0);
    std::vector<llvm::Value *> vec4s;
    for (int i = 0; i < (int)constOffsets.size(); i += 4)
        vec4s.push_back(lAssemble4Vector(loadOps, &constOffsets[i], insertBefore));

    // And now concatenate 1, 2, or 4 of the 4-wide vectors computed above
    // into 4, 8, or 16-wide final result vectors.
    int numGathers = constOffsets.size() / g->target->getVectorWidth();
    for (int i = 0; i < numGathers; ++i) {
        llvm::Value *result = NULL;
        switch (g->target->getVectorWidth()) {
        case 4:
            result = vec4s[i];
            break;
        case 8:
            result = LLVMConcatVectors(vec4s[2 * i], vec4s[2 * i + 1], insertBefore);
            break;
        case 16: {
            llvm::Value *v1 = LLVMConcatVectors(vec4s[4 * i], vec4s[4 * i + 1], insertBefore);
            llvm::Value *v2 = LLVMConcatVectors(vec4s[4 * i + 2], vec4s[4 * i + 3], insertBefore);
            result = LLVMConcatVectors(v1, v2, insertBefore);
            break;
        }
        default:
            FATAL("Unhandled vector width in lAssembleResultVectors()");
        }

        results.push_back(result);
    }
}

/** Given a call to a gather function, extract the base pointer, the 2/4/8
    scale, and the first varying offsets value to use them to compute that
    scalar base pointer that is shared by all of the gathers in the group.
    (Thus, this base pointer plus the constant offsets term for each gather
    gives the set of addresses to use for each gather.
 */
static llvm::Value *lComputeBasePtr(llvm::CallInst *gatherInst, llvm::Type *baseType, llvm::Instruction *insertBefore) {
    llvm::Value *basePtr = gatherInst->getArgOperand(0);
    llvm::Value *variableOffsets = gatherInst->getArgOperand(1);
    llvm::Value *offsetScale = gatherInst->getArgOperand(2);
    // All of the variable offsets values should be the same, due to
    // checking for this in GatherCoalescePass::runOnBasicBlock().  Thus,
    // extract the first value and use that as a scalar.
    llvm::Value *variable = LLVMExtractFirstVectorElement(variableOffsets);
    Assert(variable != NULL);
    if (variable->getType() == LLVMTypes::Int64Type)
        offsetScale = new llvm::ZExtInst(offsetScale, LLVMTypes::Int64Type, "scale_to64", insertBefore);
    llvm::Value *offset =
        llvm::BinaryOperator::Create(llvm::Instruction::Mul, variable, offsetScale, "offset", insertBefore);

    return LLVMGEPInst(basePtr, baseType, offset, "new_base", insertBefore);
}

/** Extract the constant offsets (from the common base pointer) from each
    of the gathers in a set to be coalesced.  These come in as byte
    offsets, but we'll transform them into offsets in terms of the size of
    the base scalar type being gathered.  (e.g. for an i32 gather, we might
    have offsets like <0,4,16,20>, which would be transformed to <0,1,4,5>
    here.)
 */
static void lExtractConstOffsets(const std::vector<llvm::CallInst *> &coalesceGroup, int elementSize,
                                 std::vector<int64_t> *constOffsets) {
    int width = g->target->getVectorWidth();
    *constOffsets = std::vector<int64_t>(coalesceGroup.size() * width, 0);

    int64_t *endPtr = &((*constOffsets)[0]);
    for (int i = 0; i < (int)coalesceGroup.size(); ++i, endPtr += width) {
        llvm::Value *offsets = coalesceGroup[i]->getArgOperand(3);
        int nElts;
        bool ok = LLVMExtractVectorInts(offsets, endPtr, &nElts);
        Assert(ok && nElts == width);
    }

    for (int i = 0; i < (int)constOffsets->size(); ++i)
        (*constOffsets)[i] /= elementSize;
}

/** Actually do the coalescing.  We have a set of gathers all accessing
    addresses of the form:

    (ptr + {1,2,4,8} * varyingOffset) + constOffset, a.k.a.
    basePtr + constOffset

    where varyingOffset actually has the same value across all of the SIMD
    lanes and where the part in parenthesis has the same value for all of
    the gathers in the group.
 */
static bool lCoalesceGathers(const std::vector<llvm::CallInst *> &coalesceGroup, llvm::Type *baseType) {
    llvm::Instruction *insertBefore = coalesceGroup[0];

    // First, compute the shared base pointer for all of the gathers
    llvm::Value *basePtr = lComputeBasePtr(coalesceGroup[0], baseType, insertBefore);

    int elementSize = 0;
    if (coalesceGroup[0]->getType() == LLVMTypes::Int32VectorType ||
        coalesceGroup[0]->getType() == LLVMTypes::FloatVectorType)
        elementSize = 4;
    else if (coalesceGroup[0]->getType() == LLVMTypes::Int64VectorType ||
             coalesceGroup[0]->getType() == LLVMTypes::DoubleVectorType)
        elementSize = 8;
    else
        FATAL("Unexpected gather type in lCoalesceGathers");

    // Extract the constant offsets from the gathers into the constOffsets
    // vector: the first vectorWidth elements will be those for the first
    // gather, the next vectorWidth those for the next gather, and so
    // forth.
    std::vector<int64_t> constOffsets;
    lExtractConstOffsets(coalesceGroup, elementSize, &constOffsets);

    // Determine a set of loads to perform to get all of the values we need
    // loaded.
    std::vector<CoalescedLoadOp> loadOps;
    lSelectLoads(constOffsets, &loadOps);

    lCoalescePerfInfo(coalesceGroup, loadOps);

    // Actually emit load instructions for them
    lEmitLoads(basePtr, baseType, loadOps, elementSize, insertBefore);

    // Now, for any loads that give us <8 x i32> vectors, split their
    // values into two <4 x i32> vectors; it turns out that LLVM gives us
    // better code on AVX when we assemble the pieces from 4-wide vectors.
    loadOps = lSplit8WideLoads(loadOps, insertBefore);

    // Given all of these chunks of values, shuffle together a vector that
    // gives us each result value; the i'th element of results[] gives the
    // result for the i'th gather in coalesceGroup.
    std::vector<llvm::Value *> results;
    lAssembleResultVectors(loadOps, constOffsets, results, insertBefore);

    // Finally, replace each of the original gathers with the instruction
    // that gives the value from the coalescing process.
    Assert(results.size() == coalesceGroup.size());
    for (int i = 0; i < (int)results.size(); ++i) {
        llvm::Instruction *ir = llvm::dyn_cast<llvm::Instruction>(results[i]);
        Assert(ir != NULL);

        llvm::Type *origType = coalesceGroup[i]->getType();
        if (origType != ir->getType())
            ir = new llvm::BitCastInst(ir, origType, ir->getName(), coalesceGroup[i]);

        // Previously, all of the instructions to compute the final result
        // were into the basic block here; here we remove the very last one
        // of them (that holds the final result) from the basic block.
        // This way, the following ReplaceInstWithInst() call will operate
        // successfully. (It expects that the second argument not be in any
        // basic block.)
        ir->removeFromParent();

        llvm::ReplaceInstWithInst(coalesceGroup[i], ir);
    }

    return true;
}

/** Given an instruction, returns true if the instructon may write to
    memory.  This is a conservative test in that it may return true for
    some instructions that don't actually end up writing to memory, but
    should never return false for an instruction that does write to
    memory. */
static bool lInstructionMayWriteToMemory(llvm::Instruction *inst) {
    if (llvm::isa<llvm::StoreInst>(inst) || llvm::isa<llvm::AtomicRMWInst>(inst) ||
        llvm::isa<llvm::AtomicCmpXchgInst>(inst))
        // FIXME: we could be less conservative and try to allow stores if
        // we are sure that the pointers don't overlap..
        return true;

    // Otherwise, any call instruction that doesn't have an attribute
    // indicating it won't write to memory has to be treated as a potential
    // store.
    llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst);
    if (ci != NULL) {
        llvm::Function *calledFunc = ci->getCalledFunction();
        if (calledFunc == NULL)
            return true;

        if (calledFunc->onlyReadsMemory() || calledFunc->doesNotAccessMemory())
            return false;
        return true;
    }

    return false;
}

bool GatherCoalescePass::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("GatherCoalescePass");

    llvm::Function *gatherFuncs[] = {
        m->module->getFunction("__pseudo_gather_factored_base_offsets32_i32"),
        m->module->getFunction("__pseudo_gather_factored_base_offsets32_float"),
        m->module->getFunction("__pseudo_gather_factored_base_offsets64_i32"),
        m->module->getFunction("__pseudo_gather_factored_base_offsets64_float"),
    };
    int nGatherFuncs = sizeof(gatherFuncs) / sizeof(gatherFuncs[0]);

    bool modifiedAny = false;

restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        // Iterate over all of the instructions and look for calls to
        // __pseudo_gather_factored_base_offsets{32,64}_{i32,float} calls.
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst == NULL)
            continue;

        llvm::Function *calledFunc = callInst->getCalledFunction();
        if (calledFunc == NULL)
            continue;

        int i;
        for (i = 0; i < nGatherFuncs; ++i)
            if (gatherFuncs[i] != NULL && calledFunc == gatherFuncs[i])
                break;
        if (i == nGatherFuncs)
            // Doesn't match any of the types of gathers we care about
            continue;

        SourcePos pos;
        LLVMGetSourcePosFromMetadata(callInst, &pos);
        Debug(pos, "Checking for coalescable gathers starting here...");

        llvm::Value *base = callInst->getArgOperand(0);
        llvm::Value *variableOffsets = callInst->getArgOperand(1);
        llvm::Value *offsetScale = callInst->getArgOperand(2);
        llvm::Value *mask = callInst->getArgOperand(4);

        // To apply this optimization, we need a set of one or more gathers
        // that fulfill the following conditions:
        //
        // - Mask all on
        // - The variable offsets to all have the same value (i.e., to be
        //   uniform).
        // - Same base pointer, variable offsets, and offset scale (for
        //   more than one gather)
        //
        // Then and only then do we have a common base pointer with all
        // offsets from that constants (in which case we can potentially
        // coalesce).
        if (GetMaskStatusFromValue(mask) != MaskStatus::all_on)
            continue;

        if (!LLVMVectorValuesAllEqual(variableOffsets))
            continue;

        // coalesceGroup stores the set of gathers that we're going to try to
        // coalesce over
        std::vector<llvm::CallInst *> coalesceGroup;
        coalesceGroup.push_back(callInst);

        // Start iterating at the instruction after the initial gather;
        // look at the remainder of instructions in the basic block (up
        // until we reach a write to memory) to try to find any other
        // gathers that can coalesce with this one.
        llvm::BasicBlock::iterator fwdIter = iter;
        ++fwdIter;
        for (; fwdIter != bb.end(); ++fwdIter) {
            // Must stop once we come to an instruction that may write to
            // memory; otherwise we could end up moving a read before this
            // write.
            if (lInstructionMayWriteToMemory(&*fwdIter))
                break;

            llvm::CallInst *fwdCall = llvm::dyn_cast<llvm::CallInst>(&*fwdIter);
            if (fwdCall == NULL || fwdCall->getCalledFunction() != calledFunc)
                continue;

            SourcePos fwdPos;
            // TODO: need to redesign metadata attached to pseudo calls,
            // LLVM drops metadata frequently and it results in bad disgnostics.
            LLVMGetSourcePosFromMetadata(fwdCall, &fwdPos);

#ifndef ISPC_NO_DUMPS
            if (g->debugPrint) {
                if (base != fwdCall->getArgOperand(0)) {
                    Debug(fwdPos, "base pointers mismatch");
                    LLVMDumpValue(base);
                    LLVMDumpValue(fwdCall->getArgOperand(0));
                }
                if (variableOffsets != fwdCall->getArgOperand(1)) {
                    Debug(fwdPos, "varying offsets mismatch");
                    LLVMDumpValue(variableOffsets);
                    LLVMDumpValue(fwdCall->getArgOperand(1));
                }
                if (offsetScale != fwdCall->getArgOperand(2)) {
                    Debug(fwdPos, "offset scales mismatch");
                    LLVMDumpValue(offsetScale);
                    LLVMDumpValue(fwdCall->getArgOperand(2));
                }
                if (mask != fwdCall->getArgOperand(4)) {
                    Debug(fwdPos, "masks mismatch");
                    LLVMDumpValue(mask);
                    LLVMDumpValue(fwdCall->getArgOperand(4));
                }
            }
#endif

            if (base == fwdCall->getArgOperand(0) && variableOffsets == fwdCall->getArgOperand(1) &&
                offsetScale == fwdCall->getArgOperand(2) && mask == fwdCall->getArgOperand(4)) {
                Debug(fwdPos, "This gather can be coalesced.");
                coalesceGroup.push_back(fwdCall);

                if (coalesceGroup.size() == 4)
                    // FIXME: untested heuristic: don't try to coalesce
                    // over a window of more than 4 gathers, so that we
                    // don't cause too much register pressure and end up
                    // spilling to memory anyway.
                    break;
            } else
                Debug(fwdPos, "This gather doesn't match the initial one.");
        }

        Debug(pos, "Done with checking for matching gathers");

        // Now that we have a group of gathers, see if we can coalesce them
        // into something more efficient than the original set of gathers.
        if (lCoalesceGathers(coalesceGroup, baseType)) {
            modifiedAny = true;
            goto restart;
        }
    }

    DEBUG_END_PASS("GatherCoalescePass");

    return modifiedAny;
}

bool GatherCoalescePass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("GatherCoalescePass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateGatherCoalescePass() { return new GatherCoalescePass; }

///////////////////////////////////////////////////////////////////////////
// IsCompileTimeConstantPass

/** LLVM IR implementations of target-specific functions may include calls
    to the functions "bool __is_compile_time_constant_*(...)"; these allow
    them to have specialied code paths for where the corresponding value is
    known at compile time.  For masks, for example, this allows them to not
    incur the cost of a MOVMSK call at runtime to compute its value in
    cases where the mask value isn't known until runtime.

    This pass resolves these calls into either 'true' or 'false' values so
    that later optimization passes can operate with these as constants.

    See stdlib.m4 for a number of uses of this idiom.
 */

class IsCompileTimeConstantPass : public llvm::FunctionPass {
  public:
    static char ID;
    IsCompileTimeConstantPass(bool last = false) : FunctionPass(ID) { isLastTry = last; }

    llvm::StringRef getPassName() const { return "Resolve \"is compile time constant\""; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);

    bool isLastTry;
};

char IsCompileTimeConstantPass::ID = 0;

bool IsCompileTimeConstantPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("IsCompileTimeConstantPass");

    llvm::Function *funcs[] = {m->module->getFunction("__is_compile_time_constant_mask"),
                               m->module->getFunction("__is_compile_time_constant_uniform_int32"),
                               m->module->getFunction("__is_compile_time_constant_varying_int32")};

    bool modifiedAny = false;
restart:
    for (llvm::BasicBlock::iterator i = bb.begin(), e = bb.end(); i != e; ++i) {
        // Iterate through the instructions looking for calls to the
        // __is_compile_time_constant_*() functions
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*i);
        if (callInst == NULL)
            continue;

        int j;
        int nFuncs = sizeof(funcs) / sizeof(funcs[0]);
        for (j = 0; j < nFuncs; ++j) {
            if (funcs[j] != NULL && callInst->getCalledFunction() == funcs[j])
                break;
        }
        if (j == nFuncs)
            // not a __is_compile_time_constant_* function
            continue;

        // This optimization pass can be disabled with both the (poorly
        // named) disableGatherScatterFlattening option and
        // disableMaskAllOnOptimizations.
        if (g->opt.disableGatherScatterFlattening || g->opt.disableMaskAllOnOptimizations) {
            llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, LLVMFalse);
            modifiedAny = true;
            goto restart;
        }

        // Is it a constant?  Bingo, turn the call's value into a constant
        // true value.
        llvm::Value *operand = callInst->getArgOperand(0);
        if (llvm::isa<llvm::Constant>(operand)) {
            llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, LLVMTrue);
            modifiedAny = true;
            goto restart;
        }

        // This pass runs multiple times during optimization.  Up until the
        // very last time, it only replaces the call with a 'true' if the
        // value is known to be constant and otherwise leaves the call
        // alone, in case further optimization passes can help resolve its
        // value.  The last time through, it eventually has to give up, and
        // replaces any remaining ones with 'false' constants.
        if (isLastTry) {
            llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, LLVMFalse);
            modifiedAny = true;
            goto restart;
        }
    }

    DEBUG_END_PASS("IsCompileTimeConstantPass");

    return modifiedAny;
}

bool IsCompileTimeConstantPass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("IsCompileTimeConstantPass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateIsCompileTimeConstantPass(bool isLastTry) { return new IsCompileTimeConstantPass(isLastTry); }

//////////////////////////////////////////////////////////////////////////
// DebugPass

/** This pass is added in list of passes after optimizations which
    we want to debug and print dump of LLVM IR in stderr. Also it
    prints name and number of previous optimization.
 */
#ifndef ISPC_NO_DUMPS
class DebugPass : public llvm::ModulePass {
  public:
    static char ID;
    DebugPass(char *output) : ModulePass(ID) { snprintf(str_output, sizeof(str_output), "%s", output); }

    llvm::StringRef getPassName() const { return "Dump LLVM IR"; }
    bool runOnModule(llvm::Module &m);

  private:
    char str_output[100];
};

char DebugPass::ID = 0;

bool DebugPass::runOnModule(llvm::Module &module) {
    fprintf(stderr, "%s", str_output);
    fflush(stderr);
    module.dump();
    return true;
}

static llvm::Pass *CreateDebugPass(char *output) { return new DebugPass(output); }
#endif

//////////////////////////////////////////////////////////////////////////
// DebugPassFile

/** This pass is added in list of passes after optimizations which
    we want to debug and print dump of LLVM IR to file.
 */
#ifndef ISPC_NO_DUMPS
class DebugPassFile : public llvm::ModulePass {
  public:
    static char ID;
    DebugPassFile(int number, llvm::StringRef name, std::string dir)
        : ModulePass(ID), pnum(number), pname(name), pdir(dir) {}

    llvm::StringRef getPassName() const { return "Dump LLVM IR"; }
    bool runOnModule(llvm::Module &m);
    bool doInitialization(llvm::Module &m);

  private:
    void run(llvm::Module &m, bool init);
    int pnum;
    llvm::StringRef pname;
    std::string pdir;
};

char DebugPassFile::ID = 0;

/**
 * Strips all non-alphanumeric characters from given string.
 */
std::string sanitize(std::string in) {
    llvm::Regex r("[^[:alnum:]]");
    while (r.match(in))
        in = r.sub("", in);
    return in;
}

void DebugPassFile::run(llvm::Module &module, bool init) {
    std::ostringstream oss;
    oss << (init ? "init_" : "ir_") << pnum << "_" << sanitize(std::string{pname}) << ".ll";

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

bool DebugPassFile::runOnModule(llvm::Module &module) {
    run(module, false);
    return true;
}

bool DebugPassFile::doInitialization(llvm::Module &module) {
    run(module, true);
    return true;
}

static llvm::Pass *CreateDebugPassFile(int number, llvm::StringRef name, std::string dir) {
    return new DebugPassFile(number, name, dir);
}
#endif

///////////////////////////////////////////////////////////////////////////
// MakeInternalFuncsStaticPass

/** There are a number of target-specific functions that we use during
    these optimization passes.  By the time we are done with optimization,
    any uses of these should be inlined and no calls to these functions
    should remain.  This pass marks all of these functions as having
    private linkage so that subsequent passes can eliminate them as dead
    code, thus cleaning up the final code output by the compiler.  We can't
    just declare these as static from the start, however, since then they
    end up being eliminated as dead code during early optimization passes
    even though we may need to generate calls to them during later
    optimization passes.
 */
class MakeInternalFuncsStaticPass : public llvm::ModulePass {
  public:
    static char ID;
    MakeInternalFuncsStaticPass(bool last = false) : ModulePass(ID) {}

    void getAnalysisUsage(llvm::AnalysisUsage &AU) const { AU.setPreservesCFG(); }

    llvm::StringRef getPassName() const { return "Make internal funcs \"static\""; }
    bool runOnModule(llvm::Module &m);
};

char MakeInternalFuncsStaticPass::ID = 0;

bool MakeInternalFuncsStaticPass::runOnModule(llvm::Module &module) {
    const char *names[] = {
        "__avg_up_uint8",
        "__avg_up_int8",
        "__avg_up_uint16",
        "__avg_up_int16",
        "__avg_down_uint8",
        "__avg_down_int8",
        "__avg_down_uint16",
        "__avg_down_int16",
        "__fast_masked_vload",
        "__gather_factored_base_offsets32_i8",
        "__gather_factored_base_offsets32_i16",
        "__gather_factored_base_offsets32_i32",
        "__gather_factored_base_offsets32_i64",
        "__gather_factored_base_offsets32_half",
        "__gather_factored_base_offsets32_float",
        "__gather_factored_base_offsets32_double",
        "__gather_factored_base_offsets64_i8",
        "__gather_factored_base_offsets64_i16",
        "__gather_factored_base_offsets64_i32",
        "__gather_factored_base_offsets64_i64",
        "__gather_factored_base_offsets64_half",
        "__gather_factored_base_offsets64_float",
        "__gather_factored_base_offsets64_double",
        "__gather_base_offsets32_i8",
        "__gather_base_offsets32_i16",
        "__gather_base_offsets32_i32",
        "__gather_base_offsets32_i64",
        "__gather_base_offsets32_half",
        "__gather_base_offsets32_float",
        "__gather_base_offsets32_double",
        "__gather_base_offsets64_i8",
        "__gather_base_offsets64_i16",
        "__gather_base_offsets64_i32",
        "__gather_base_offsets64_i64",
        "__gather_base_offsets64_half",
        "__gather_base_offsets64_float",
        "__gather_base_offsets64_double",
        "__gather32_i8",
        "__gather32_i16",
        "__gather32_i32",
        "__gather32_i64",
        "__gather32_half",
        "__gather32_float",
        "__gather32_double",
        "__gather64_i8",
        "__gather64_i16",
        "__gather64_i32",
        "__gather64_i64",
        "__gather64_half",
        "__gather64_float",
        "__gather64_double",
        "__gather_elt32_i8",
        "__gather_elt32_i16",
        "__gather_elt32_i32",
        "__gather_elt32_i64",
        "__gather_elt32_half",
        "__gather_elt32_float",
        "__gather_elt32_double",
        "__gather_elt64_i8",
        "__gather_elt64_i16",
        "__gather_elt64_i32",
        "__gather_elt64_i64",
        "__gather_elt64_half",
        "__gather_elt64_float",
        "__gather_elt64_double",
        "__masked_load_i8",
        "__masked_load_i16",
        "__masked_load_i32",
        "__masked_load_i64",
        "__masked_load_half",
        "__masked_load_float",
        "__masked_load_double",
        "__masked_store_i8",
        "__masked_store_i16",
        "__masked_store_i32",
        "__masked_store_i64",
        "__masked_store_half",
        "__masked_store_float",
        "__masked_store_double",
        "__masked_store_blend_i8",
        "__masked_store_blend_i16",
        "__masked_store_blend_i32",
        "__masked_store_blend_i64",
        "__masked_store_blend_half",
        "__masked_store_blend_float",
        "__masked_store_blend_double",
        "__scatter_factored_base_offsets32_i8",
        "__scatter_factored_base_offsets32_i16",
        "__scatter_factored_base_offsets32_i32",
        "__scatter_factored_base_offsets32_i64",
        "__scatter_factored_base_offsets32_half",
        "__scatter_factored_base_offsets32_float",
        "__scatter_factored_base_offsets32_double",
        "__scatter_factored_base_offsets64_i8",
        "__scatter_factored_base_offsets64_i16",
        "__scatter_factored_base_offsets64_i32",
        "__scatter_factored_base_offsets64_i64",
        "__scatter_factored_base_offsets64_half",
        "__scatter_factored_base_offsets64_float",
        "__scatter_factored_base_offsets64_double",
        "__scatter_base_offsets32_i8",
        "__scatter_base_offsets32_i16",
        "__scatter_base_offsets32_i32",
        "__scatter_base_offsets32_i64",
        "__scatter_base_offsets32_half",
        "__scatter_base_offsets32_float",
        "__scatter_base_offsets32_double",
        "__scatter_base_offsets64_i8",
        "__scatter_base_offsets64_i16",
        "__scatter_base_offsets64_i32",
        "__scatter_base_offsets64_i64",
        "__scatter_base_offsets64_half",
        "__scatter_base_offsets64_float",
        "__scatter_base_offsets64_double",
        "__scatter_elt32_i8",
        "__scatter_elt32_i16",
        "__scatter_elt32_i32",
        "__scatter_elt32_i64",
        "__scatter_elt32_half",
        "__scatter_elt32_float",
        "__scatter_elt32_double",
        "__scatter_elt64_i8",
        "__scatter_elt64_i16",
        "__scatter_elt64_i32",
        "__scatter_elt64_i64",
        "__scatter_elt64_half",
        "__scatter_elt64_float",
        "__scatter_elt64_double",
        "__scatter32_i8",
        "__scatter32_i16",
        "__scatter32_i32",
        "__scatter32_i64",
        "__scatter32_half",
        "__scatter32_float",
        "__scatter32_double",
        "__scatter64_i8",
        "__scatter64_i16",
        "__scatter64_i32",
        "__scatter64_i64",
        "__scatter64_half",
        "__scatter64_float",
        "__scatter64_double",
        "__prefetch_read_varying_1",
        "__prefetch_read_varying_2",
        "__prefetch_read_varying_3",
        "__prefetch_read_varying_nt",
        "__prefetch_write_varying_1",
        "__prefetch_write_varying_2",
        "__prefetch_write_varying_3",
        "__keep_funcs_live",
#ifdef ISPC_XE_ENABLED
        "__masked_load_blend_i8",
        "__masked_load_blend_i16",
        "__masked_load_blend_i32",
        "__masked_load_blend_i64",
        "__masked_load_blend_half",
        "__masked_load_blend_float",
        "__masked_load_blend_double",
#endif
    };

    bool modifiedAny = false;
    int count = sizeof(names) / sizeof(names[0]);
    for (int i = 0; i < count; ++i) {
        llvm::Function *f = m->module->getFunction(names[i]);
        if (f != NULL && f->empty() == false) {
            f->setLinkage(llvm::GlobalValue::InternalLinkage);
            modifiedAny = true;
        }
    }

    return modifiedAny;
}

static llvm::Pass *CreateMakeInternalFuncsStaticPass() { return new MakeInternalFuncsStaticPass; }

///////////////////////////////////////////////////////////////////////////
// PeepholePass

class PeepholePass : public llvm::FunctionPass {
  public:
    PeepholePass();

    llvm::StringRef getPassName() const { return "Peephole Optimizations"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);

    static char ID;
};

char PeepholePass::ID = 0;

PeepholePass::PeepholePass() : FunctionPass(ID) {}

using namespace llvm::PatternMatch;

template <typename Op_t, unsigned Opcode> struct CastClassTypes_match {
    Op_t Op;
    const llvm::Type *fromType, *toType;

    CastClassTypes_match(const Op_t &OpMatch, const llvm::Type *f, const llvm::Type *t)
        : Op(OpMatch), fromType(f), toType(t) {}

    template <typename OpTy> bool match(OpTy *V) {
        if (llvm::Operator *O = llvm::dyn_cast<llvm::Operator>(V))
            return (O->getOpcode() == Opcode && Op.match(O->getOperand(0)) && O->getType() == toType &&
                    O->getOperand(0)->getType() == fromType);
        return false;
    }
};

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::SExt> m_SExt8To16(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::SExt>(Op, LLVMTypes::Int8VectorType,
                                                               LLVMTypes::Int16VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::ZExt> m_ZExt8To16(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::ZExt>(Op, LLVMTypes::Int8VectorType,
                                                               LLVMTypes::Int16VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::Trunc> m_Trunc16To8(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::Trunc>(Op, LLVMTypes::Int16VectorType,
                                                                LLVMTypes::Int8VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::SExt> m_SExt16To32(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::SExt>(Op, LLVMTypes::Int16VectorType,
                                                               LLVMTypes::Int32VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::ZExt> m_ZExt16To32(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::ZExt>(Op, LLVMTypes::Int16VectorType,
                                                               LLVMTypes::Int32VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::Trunc> m_Trunc32To16(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::Trunc>(Op, LLVMTypes::Int32VectorType,
                                                                LLVMTypes::Int16VectorType);
}

template <typename Op_t> struct UDiv2_match {
    Op_t Op;

    UDiv2_match(const Op_t &OpMatch) : Op(OpMatch) {}

    template <typename OpTy> bool match(OpTy *V) {
        llvm::BinaryOperator *bop;
        llvm::ConstantDataVector *cdv;
        if ((bop = llvm::dyn_cast<llvm::BinaryOperator>(V)) &&
            (cdv = llvm::dyn_cast<llvm::ConstantDataVector>(bop->getOperand(1))) && cdv->getSplatValue() != NULL) {
            const llvm::APInt &apInt = cdv->getUniqueInteger();

            switch (bop->getOpcode()) {
            case llvm::Instruction::UDiv:
                // divide by 2
                return (apInt.isIntN(2) && Op.match(bop->getOperand(0)));
            case llvm::Instruction::LShr:
                // shift left by 1
                return (apInt.isIntN(1) && Op.match(bop->getOperand(0)));
            default:
                return false;
            }
        }
        return false;
    }
};

template <typename V> inline UDiv2_match<V> m_UDiv2(const V &v) { return UDiv2_match<V>(v); }

template <typename Op_t> struct SDiv2_match {
    Op_t Op;

    SDiv2_match(const Op_t &OpMatch) : Op(OpMatch) {}

    template <typename OpTy> bool match(OpTy *V) {
        llvm::BinaryOperator *bop;
        llvm::ConstantDataVector *cdv;
        if ((bop = llvm::dyn_cast<llvm::BinaryOperator>(V)) &&
            (cdv = llvm::dyn_cast<llvm::ConstantDataVector>(bop->getOperand(1))) && cdv->getSplatValue() != NULL) {
            const llvm::APInt &apInt = cdv->getUniqueInteger();

            switch (bop->getOpcode()) {
            case llvm::Instruction::SDiv:
                // divide by 2
                return (apInt.isIntN(2) && Op.match(bop->getOperand(0)));
            case llvm::Instruction::AShr:
                // shift left by 1
                return (apInt.isIntN(1) && Op.match(bop->getOperand(0)));
            default:
                return false;
            }
        }
        return false;
    }
};

template <typename V> inline SDiv2_match<V> m_SDiv2(const V &v) { return SDiv2_match<V>(v); }

// Returns true if the given function has a call to an intrinsic function
// in its definition.
static bool lHasIntrinsicInDefinition(llvm::Function *func) {
    llvm::Function::iterator bbiter = func->begin();
    for (; bbiter != func->end(); ++bbiter) {
        for (llvm::BasicBlock::iterator institer = bbiter->begin(); institer != bbiter->end(); ++institer) {
            if (llvm::isa<llvm::IntrinsicInst>(institer))
                return true;
        }
    }
    return false;
}

static llvm::Instruction *lGetBinaryIntrinsic(const char *name, llvm::Value *opa, llvm::Value *opb) {
    llvm::Function *func = m->module->getFunction(name);
    Assert(func != NULL);

    // Make sure that the definition of the llvm::Function has a call to an
    // intrinsic function in its instructions; otherwise we will generate
    // infinite loops where we "helpfully" turn the default implementations
    // of target builtins like __avg_up_uint8 that are implemented with plain
    // arithmetic ops into recursive calls to themselves.
    if (lHasIntrinsicInDefinition(func))
        return LLVMCallInst(func, opa, opb, name);
    else
        return NULL;
}

//////////////////////////////////////////////////

static llvm::Instruction *lMatchAvgUpUInt8(llvm::Value *inst) {
    // (unsigned int8)(((unsigned int16)a + (unsigned int16)b + 1)/2)
    llvm::Value *opa, *opb;
    const llvm::APInt *delta;
    if (match(inst, m_Trunc16To8(m_UDiv2(m_CombineOr(
                        m_CombineOr(m_Add(m_ZExt8To16(m_Value(opa)), m_Add(m_ZExt8To16(m_Value(opb)), m_APInt(delta))),
                                    m_Add(m_Add(m_ZExt8To16(m_Value(opa)), m_APInt(delta)), m_ZExt8To16(m_Value(opb)))),
                        m_Add(m_Add(m_ZExt8To16(m_Value(opa)), m_ZExt8To16(m_Value(opb))), m_APInt(delta))))))) {
        if (delta->isIntN(1) == false)
            return NULL;

        return lGetBinaryIntrinsic("__avg_up_uint8", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgDownUInt8(llvm::Value *inst) {
    // (unsigned int8)(((unsigned int16)a + (unsigned int16)b)/2)
    llvm::Value *opa, *opb;
    if (match(inst, m_Trunc16To8(m_UDiv2(m_Add(m_ZExt8To16(m_Value(opa)), m_ZExt8To16(m_Value(opb))))))) {
        return lGetBinaryIntrinsic("__avg_down_uint8", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgUpUInt16(llvm::Value *inst) {
    // (unsigned int16)(((unsigned int32)a + (unsigned int32)b + 1)/2)
    llvm::Value *opa, *opb;
    const llvm::APInt *delta;
    if (match(inst,
              m_Trunc32To16(m_UDiv2(m_CombineOr(
                  m_CombineOr(m_Add(m_ZExt16To32(m_Value(opa)), m_Add(m_ZExt16To32(m_Value(opb)), m_APInt(delta))),
                              m_Add(m_Add(m_ZExt16To32(m_Value(opa)), m_APInt(delta)), m_ZExt16To32(m_Value(opb)))),
                  m_Add(m_Add(m_ZExt16To32(m_Value(opa)), m_ZExt16To32(m_Value(opb))), m_APInt(delta))))))) {
        if (delta->isIntN(1) == false)
            return NULL;

        return lGetBinaryIntrinsic("__avg_up_uint16", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgDownUInt16(llvm::Value *inst) {
    // (unsigned int16)(((unsigned int32)a + (unsigned int32)b)/2)
    llvm::Value *opa, *opb;
    if (match(inst, m_Trunc32To16(m_UDiv2(m_Add(m_ZExt16To32(m_Value(opa)), m_ZExt16To32(m_Value(opb))))))) {
        return lGetBinaryIntrinsic("__avg_down_uint16", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgUpInt8(llvm::Value *inst) {
    // (int8)(((int16)a + (int16)b + 1)/2)
    llvm::Value *opa, *opb;
    const llvm::APInt *delta;
    if (match(inst, m_Trunc16To8(m_SDiv2(m_CombineOr(
                        m_CombineOr(m_Add(m_SExt8To16(m_Value(opa)), m_Add(m_SExt8To16(m_Value(opb)), m_APInt(delta))),
                                    m_Add(m_Add(m_SExt8To16(m_Value(opa)), m_APInt(delta)), m_SExt8To16(m_Value(opb)))),
                        m_Add(m_Add(m_SExt8To16(m_Value(opa)), m_SExt8To16(m_Value(opb))), m_APInt(delta))))))) {
        if (delta->isIntN(1) == false)
            return NULL;

        return lGetBinaryIntrinsic("__avg_up_int8", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgDownInt8(llvm::Value *inst) {
    // (int8)(((int16)a + (int16)b)/2)
    llvm::Value *opa, *opb;
    if (match(inst, m_Trunc16To8(m_SDiv2(m_Add(m_SExt8To16(m_Value(opa)), m_SExt8To16(m_Value(opb))))))) {
        return lGetBinaryIntrinsic("__avg_down_int8", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgUpInt16(llvm::Value *inst) {
    // (int16)(((int32)a + (int32)b + 1)/2)
    llvm::Value *opa, *opb;
    const llvm::APInt *delta;
    if (match(inst,
              m_Trunc32To16(m_SDiv2(m_CombineOr(
                  m_CombineOr(m_Add(m_SExt16To32(m_Value(opa)), m_Add(m_SExt16To32(m_Value(opb)), m_APInt(delta))),
                              m_Add(m_Add(m_SExt16To32(m_Value(opa)), m_APInt(delta)), m_SExt16To32(m_Value(opb)))),
                  m_Add(m_Add(m_SExt16To32(m_Value(opa)), m_SExt16To32(m_Value(opb))), m_APInt(delta))))))) {
        if (delta->isIntN(1) == false)
            return NULL;

        return lGetBinaryIntrinsic("__avg_up_int16", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgDownInt16(llvm::Value *inst) {
    // (int16)(((int32)a + (int32)b)/2)
    llvm::Value *opa, *opb;
    if (match(inst, m_Trunc32To16(m_SDiv2(m_Add(m_SExt16To32(m_Value(opa)), m_SExt16To32(m_Value(opb))))))) {
        return lGetBinaryIntrinsic("__avg_down_int16", opa, opb);
    }
    return NULL;
}

bool PeepholePass::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("PeepholePass");

    bool modifiedAny = false;
restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::Instruction *inst = &*iter;

        llvm::Instruction *builtinCall = lMatchAvgUpUInt8(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgUpUInt16(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgDownUInt8(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgDownUInt16(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgUpInt8(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgUpInt16(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgDownInt8(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgDownInt16(inst);
        if (builtinCall != NULL) {
            llvm::ReplaceInstWithInst(inst, builtinCall);
            modifiedAny = true;
            goto restart;
        }
    }

    DEBUG_END_PASS("PeepholePass");

    return modifiedAny;
}

bool PeepholePass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("PeepholePass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreatePeepholePass() { return new PeepholePass; }

/** Given an llvm::Value known to be an integer, return its value as
    an int64_t.
*/
static int64_t lGetIntValue(llvm::Value *offset) {
    llvm::ConstantInt *intOffset = llvm::dyn_cast<llvm::ConstantInt>(offset);
    Assert(intOffset && (intOffset->getBitWidth() == 32 || intOffset->getBitWidth() == 64));
    return intOffset->getSExtValue();
}

///////////////////////////////////////////////////////////////////////////
// ReplaceStdlibShiftPass

class ReplaceStdlibShiftPass : public llvm::FunctionPass {
  public:
    static char ID;
    ReplaceStdlibShiftPass() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const { return "Resolve \"replace extract insert chains\""; }

    bool runOnBasicBlock(llvm::BasicBlock &BB);

    bool runOnFunction(llvm::Function &F);
};

char ReplaceStdlibShiftPass::ID = 0;

// This pass replaces shift() with ShuffleVector when the offset is a constant.
// rotate() which is similar in functionality has a slightly different
// implementation. This is due to LLVM(createInstructionCombiningPass)
// optimizing rotate() implementation better when similar implementations
// are used for both. This is a hack to produce similarly optimized code for
// shift.
bool ReplaceStdlibShiftPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("ReplaceStdlibShiftPass");
    bool modifiedAny = false;

    llvm::Function *shifts[6];
    shifts[0] = m->module->getFunction("shift___vytuni");
    shifts[1] = m->module->getFunction("shift___vysuni");
    shifts[2] = m->module->getFunction("shift___vyiuni");
    shifts[3] = m->module->getFunction("shift___vyIuni");
    shifts[4] = m->module->getFunction("shift___vyfuni");
    shifts[5] = m->module->getFunction("shift___vyduni");

    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::Instruction *inst = &*iter;

        if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst)) {
            llvm::Function *func = ci->getCalledFunction();
            for (int i = 0; i < 6; i++) {
                if (shifts[i] && (shifts[i] == func)) {
                    // we matched a call
                    llvm::Value *shiftedVec = ci->getArgOperand(0);
                    llvm::Value *shiftAmt = ci->getArgOperand(1);
                    if (llvm::isa<llvm::Constant>(shiftAmt)) {
                        int vectorWidth = g->target->getVectorWidth();
                        int *shuffleVals = new int[vectorWidth];
                        int shiftInt = lGetIntValue(shiftAmt);
                        for (int i = 0; i < vectorWidth; i++) {
                            int s = i + shiftInt;
                            s = (s < 0) ? vectorWidth : s;
                            s = (s >= vectorWidth) ? vectorWidth : s;
                            shuffleVals[i] = s;
                        }
                        llvm::Value *shuffleIdxs = LLVMInt32Vector(shuffleVals);
                        llvm::Value *zeroVec = llvm::ConstantAggregateZero::get(shiftedVec->getType());
                        llvm::Value *shuffle =
                            new llvm::ShuffleVectorInst(shiftedVec, zeroVec, shuffleIdxs, "vecShift", ci);
                        ci->replaceAllUsesWith(shuffle);
                        modifiedAny = true;
                        delete[] shuffleVals;
                    } else if (g->opt.level > 0) {
                        PerformanceWarning(SourcePos(), "Stdlib shift() called without constant shift amount.");
                    }
                }
            }
        }
    }

    DEBUG_END_PASS("ReplaceStdlibShiftPass");

    return modifiedAny;
}

bool ReplaceStdlibShiftPass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("ReplaceStdlibShiftPass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateReplaceStdlibShiftPass() { return new ReplaceStdlibShiftPass(); }

#ifdef ISPC_XE_ENABLED

///////////////////////////////////////////////////////////////////////////
// Xe Coalescing passes

/*
 * MemoryCoalescing: basis for store/load coalescing optimizations.
 *
 * Memory coalescing tries to merge several memory operations into
 * wider ones. There are two different flows for store and load coalescings.
 *
 * This is a function pass. Each BB is handled separately.
 *
 * The general idea of optimization is to iterate over BB and analyse pointers
 * for all optimization targets (target is an instruction that can be optimized).
 * Instructions that uses the same pointer can be coalesced.
 *
 * Each memory instruction has a pointer operand. To perform coalescing,
 * we need to determine its base pointer and offset for it. This is done
 * via recurrent algorithm. Only constant offset can be handled.
 * TODO: found in experiments: sometimes pointer is passed as null
 * and actual pointer came via calculated offset. Such situation is not
 * handled now (Target: Xe-TPM).
 *
 * Sometimies coalescing can fail in case when it is not sure if the
 * transformation is actually safe. This can happen if some "bad"
 * instructions met between optimization targets. To handle this, all
 * optimization targets are collected in several blocks. They also can be
 * treated as non-interfering fragments of BB. Each block can be
 * safely optimized. Bad instructions trigger creation of such blocks.
 * For now, stopper detection is implemented in a general way for all
 * types of coalescing. It can be changed in future.
 *
 * Basicly, implementation can treat offset in different ways. Afterall,
 * implementation builds accesses for optimized instructions. All
 * offsets are presented in bytes now.
 *
 * To create new coalescing type, one should implement several handlers.
 * More info about them below.
 *
 * There are also some helpers that are not used by optimization
 * directly, but can be used by several coalescing implementations.
 * Such helpers contains some general stuff.
 */
class MemoryCoalescing : public llvm::FunctionPass {
  protected:
    typedef int64_t OffsetT;
    typedef llvm::SmallVector<OffsetT, 16> OffsetsVecT;
    typedef llvm::SmallVector<llvm::Instruction *, 4> BlockInstsVecT;

    // Available coalescing types
    enum class MemType { OPT_LOAD, OPT_STORE };

    // Inst optimization data
    class InstData {
      public:
        // Instruction itself
        llvm::Instruction *Inst;
        // Offsets for memory accesses
        OffsetsVecT Offsets;
        // Unused for loads, stored value for stores
        llvm::Value *Val;
        InstData() = delete;
        InstData(llvm::Instruction *Inst, OffsetsVecT &Offsets, llvm::Value *Val)
            : Inst(Inst), Offsets(Offsets), Val(Val) {}
    };

    // Ptr optimization data
    class PtrData {
      public:
        // The insertion point for the ptr users.
        // All instructions will be placed before it.
        // InsertPoint should be the first instruction in block for load coalescing
        // and the last one for store coalescing. This is achieved by different
        // traversal in both types of optimization.
        llvm::Instruction *InsertPoint;
        // Data for insts that use this ptr
        std::vector<InstData> InstsData;
        PtrData() : InsertPoint(nullptr) {}
        void addInstruction(llvm::Instruction *Inst, OffsetsVecT &Offsets, llvm::Value *Val) {
            InstsData.push_back(InstData(Inst, Offsets, Val));
            if (!InsertPoint)
                InsertPoint = Inst;
        }
    };

    // Data for optimization block
    class OptBlock {
      public:
        // Data for all ptrs in the block
        std::unordered_map<llvm::Value *, PtrData> PtrsData;
        void addInstruction(llvm::Instruction *Inst, llvm::Value *Ptr, OffsetsVecT &Offsets,
                            llvm::Value *Val = nullptr) {
            PtrsData[Ptr].addInstruction(Inst, Offsets, Val);
        }
    };

    // Instructions that are possibly dead. It includes some newly created instructions that
    // are created during optimization and memory instructions that are optimized out. All
    // instructions from this list are treated as dead if they don't have any users.
    std::set<llvm::Instruction *> PossiblyDead;
    // Modification flag
    bool modifiedAny = false;
    // Address space for optimization
    // TODO: not obvious how it would work with TPM on Xe now.
    AddressSpace AddrSpace;

  private:
    // Info for complex GEPs
    struct GEPVarOffsetInfo {
        llvm::Instruction *GEP;
        llvm::User::value_op_iterator FirstConstUse;
        GEPVarOffsetInfo(llvm::Instruction *GEP) : GEP(GEP) {}
    };

    // DenseMap helper for complex GEPs
    struct DenseMapInfo {
        static inline GEPVarOffsetInfo getEmptyKey() {
            return GEPVarOffsetInfo(llvm::DenseMapInfo<llvm::Instruction *>::getEmptyKey());
        }
        static inline GEPVarOffsetInfo getTombstoneKey() {
            return GEPVarOffsetInfo(llvm::DenseMapInfo<llvm::Instruction *>::getTombstoneKey());
        }
        static inline bool isSentinel(const GEPVarOffsetInfo &Val) {
            return Val.GEP == getEmptyKey().GEP || Val.GEP == getTombstoneKey().GEP;
        }
        static unsigned getHashValue(const GEPVarOffsetInfo &Val) {
            return hash_combine_range(Val.GEP->value_op_begin(), Val.FirstConstUse);
        }
        static bool isEqual(const GEPVarOffsetInfo &LHS, const GEPVarOffsetInfo &RHS) {
            if (isSentinel(LHS) || isSentinel(RHS))
                return LHS.GEP == RHS.GEP;

            for (auto lhs_it = LHS.GEP->value_op_begin(), rhs_it = RHS.GEP->value_op_begin(), lhs_e = LHS.FirstConstUse,
                      rhs_e = RHS.FirstConstUse;
                 lhs_it != lhs_e || rhs_it != rhs_e; ++lhs_it, ++rhs_it) {
                if (lhs_it == lhs_e || rhs_it == rhs_e)
                    return false;
                if (*lhs_it != *rhs_it)
                    return false;
            }

            return true;
        }
    };

    // Helper for ptr offset analysis. It holds
    // Ptr+Offset data. IsConstantOffset is used to
    // check if optimization is appliable for such ptr.
    // Ptr can be null, in this case only Offset
    // matters. This is used for arithmetic analysis.
    // The data is consistent only if the IsConstantOffset
    // flag is true.
    struct BasePtrInfo {
        llvm::Value *Ptr;
        OffsetT Offset;
        bool IsConstantOffset;

        BasePtrInfo() : Ptr(nullptr), Offset(0), IsConstantOffset(false) {}
    };

    // Coalescing type
    const MemType OptType;
    // Cached data for visited ptr instructions
    std::unordered_map<llvm::Value *, BasePtrInfo> BasePtrInfoCache;
    // Cached data for complex GEPs partial replacements
    llvm::DenseMap<GEPVarOffsetInfo, llvm::Value *, DenseMapInfo> ComplexGEPsInfoCache;
    // Blocks to be optimized
    std::vector<OptBlock> Blocks;

    // Find base ptr and its offset
    BasePtrInfo findBasePtr(llvm::Value *PtrOperand);
    // Analyse GEP that has variable offset. This is used by findBasePtr.
    // Such GEPs can have same variable part while the final (different) part is constant.
    BasePtrInfo analyseVarOffsetGEP(llvm::GetElementPtrInst *GEP);
    // Analyse arithmetics. That allows to handle more cases when offset is calculated not via GEP.
    // TODO: not implemented now, returns result with IsConstantOffset=false.
    BasePtrInfo analyseArithmetics(llvm::BinaryOperator *Arithm);
    // Return true if Inst blocks further optimization of currently collected
    // optimization targets. This is a stopper for collecting
    // instructions.
    bool stopsOptimization(llvm::Instruction *Inst) const;
    // Add Block to blocks list and return new one.
    OptBlock finishBlock(OptBlock &Block);
    // Collect data for optimization
    void analyseInsts(llvm::BasicBlock &BB);
    // Apply coalescing
    void applyOptimization();
    // Delete dead instructions
    void deletePossiblyDeadInsts();
    // Reset all internal structures
    void clear();

  protected:
    // Initialization
    MemoryCoalescing(char &ID, MemType OptType, AddressSpace AddrSpace)
        : FunctionPass(ID), AddrSpace(AddrSpace), OptType(OptType) {}
    // Optimization runner
    bool runOnFunction(llvm::Function &Fn);

    /* ------ Handlers ------ */
    // Methods in this block are interface for different coalescing types.

    // Return true if coalescing can handle Inst.
    virtual bool isOptimizationTarget(llvm::Instruction *Inst) const = 0;
    // Return pointer value or null if there is no one. This should handle
    // all optimization targets.
    virtual llvm::Value *getPointer(llvm::Instruction *Inst) const = 0;
    // Return offset for implied values. Scatter and gathers, for example,
    // can have vectorized offset, so the result is a list. If the offset
    // is not constant, return empty list. This should handle all optimization
    // targets.
    virtual OffsetsVecT getOffset(llvm::Instruction *Inst) const = 0;
    // Return value being stored. For load coalescing, simply return null.
    // This function is not called under load coalescing. For store coalescing,
    // this should handle all optimizations target.
    virtual llvm::Value *getStoredValue(llvm::Instruction *Inst) const = 0;
    // Perform optimization on ptr data.
    virtual void optimizePtr(llvm::Value *Ptr, PtrData &PD, llvm::Instruction *InsertPoint) = 0;

    // TODO: this handler must call runOnBasicBlockImpl to run optimization.
    // This function is needed due to DEBUG_START/END_PASS logic. Maybe there is
    // a better way to solve it.
    virtual void runOnBasicBlock(llvm::BasicBlock &BB) = 0;
    void runOnBasicBlockImpl(llvm::BasicBlock &BB);

    /* ------- Helpers ------ */
    // Methods in this block are not used in optimization directly
    // and can be invoked by handlers implementations

    // Collect constant offsets from vector. If offset is not a constant, empty list
    // is returned.
    OffsetsVecT getConstOffsetFromVector(llvm::Value *VecVal) const;
    // Multiplies all elements by scale.
    void applyScale(OffsetsVecT &Offsets, OffsetT scale) const;
    // Build InsertElementInst. This is offset based insertion and it can deal with mixed types.
    llvm::Value *buildIEI(llvm::Value *InsertTo, llvm::Value *Val, OffsetT OffsetBytes,
                          llvm::Instruction *InsertBefore) const;
    // Build ExtractElementInst. This is offset based extraction and it can deal with mixed types.
    llvm::Value *buildEEI(llvm::Value *ExtractFrom, OffsetT OffsetBytes, llvm::Type *DstTy,
                          llvm::Instruction *InsertBefore) const;
    // Build Cast (BitCast or IntToPtr)
    llvm::Value *buildCast(llvm::Value *Val, llvm::Type *DstTy, llvm::Instruction *InsertBefore) const;
    // Extract element from block values. Can aggregate value from several block instructions.
    llvm::Value *extractValueFromBlock(const BlockInstsVecT &BlockInstsVec, OffsetT OffsetBytes, llvm::Type *DstTy,
                                       llvm::Instruction *InsertBefore) const;
    // Get scalar type size in bytes
    unsigned getScalarTypeSize(llvm::Type *Ty) const;
};

// Optimization runner
bool MemoryCoalescing::runOnFunction(llvm::Function &Fn) {
    llvm::TimeTraceScope FuncScope("MemoryCoalescing::runOnFunction", Fn.getName());
    for (llvm::BasicBlock &BB : Fn) {
        runOnBasicBlock(BB);
    }

    return modifiedAny;
}

// Find base pointer info for ptr operand.
MemoryCoalescing::BasePtrInfo MemoryCoalescing::findBasePtr(llvm::Value *PtrOperand) {
    // Look for previously handled value
    auto it = BasePtrInfoCache.find(PtrOperand);
    if (it != BasePtrInfoCache.end())
        return it->second;

    BasePtrInfo res;
    if (auto BCI = llvm::dyn_cast<llvm::BitCastInst>(PtrOperand)) {
        res = findBasePtr(BCI->getOperand(0));
    } else if (auto GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(PtrOperand)) {
        if (GEP->hasAllConstantIndices()) {
            // Easy case. Collect offset.
            llvm::APInt acc(g->target->is32Bit() ? 32 : 64, 0, true);
            bool checker = GEP->accumulateConstantOffset(GEP->getModule()->getDataLayout(), acc);
            Assert(checker);

            // Run this analysis for GEP's ptr operand to handle possible bypasses
            res = findBasePtr(GEP->getPointerOperand());
            res.Offset += acc.getSExtValue();
        } else {
            // Bad case. We need partialy use the data in this GEP - there might be
            // intersections with others. The handler for this case is implemented
            // in a separate method.
            res = analyseVarOffsetGEP(GEP);
        }
    } else if (auto Arithm = llvm::dyn_cast<llvm::BinaryOperator>(PtrOperand)) {
        // Handle arithmetics
        res = analyseArithmetics(Arithm);
    } else if (llvm::isa<llvm::PointerType>(PtrOperand->getType())) {
        // This block is reached when no possible bypasses are left.
        // Use this ptr as a base
        res.Ptr = PtrOperand;
        res.Offset = 0;
        res.IsConstantOffset = true;
    } else if (auto Const = llvm::dyn_cast<llvm::ConstantInt>(PtrOperand)) {
        // Constant operand - set offset.
        res.Ptr = nullptr;
        res.Offset = Const->getSExtValue();
        res.IsConstantOffset = true;
    } else {
        // That is a non-constant offset
        res.IsConstantOffset = false;
    }

    // Finally, cache result and return
    BasePtrInfoCache[PtrOperand] = res;
    return res;
}

// Analyse GEP that has some indices non-constant.
// TODO: Consider adding full chain analysis. Decided to leave this idea for now
// because it may introduce many redundant address calculations: current approach
// would copy full chain even if some common part could be calculated somewhere eariler.
MemoryCoalescing::BasePtrInfo MemoryCoalescing::analyseVarOffsetGEP(llvm::GetElementPtrInst *GEP) {
    BasePtrInfo res;
    // Find the last constant idxs: we may be able to use them as constant offset for
    // several GEPs with common pre-constant part
    auto FirstConstIdx = GEP->getNumOperands();
    for (unsigned i = FirstConstIdx - 1; i >= 0; --i) {
        if (!llvm::isa<llvm::ConstantInt>(GEP->getOperand(i)))
            break;
        FirstConstIdx = i;
    }
    // Early return in case when no constant offset was found:
    // further actions are useless.
    if (GEP->getNumOperands() == FirstConstIdx) {
        res.Ptr = GEP;
        res.Offset = 0;
        res.IsConstantOffset = true;
        return res;
    }
    // Initialize var offset ops info.
    auto FirstConstUse = GEP->value_op_begin();
    for (unsigned i = 0; i < FirstConstIdx; ++i)
        FirstConstUse++;
    GEPVarOffsetInfo GEPVarOffsetData(GEP);
    GEPVarOffsetData.FirstConstUse = FirstConstUse;
    // Try to find existing dangling PartialGEP for such Ops combination
    // or create new one.
    auto DanglingGEP_it = ComplexGEPsInfoCache.find(GEPVarOffsetData);
    if (DanglingGEP_it == ComplexGEPsInfoCache.end()) {
        std::vector<llvm::Value *> PartialIdxs;
        for (unsigned i = 1; i < FirstConstIdx; ++i)
            PartialIdxs.push_back(GEP->getOperand(i));
        llvm::Value *tPtr = GEP->getPointerOperand();
        llvm::Type *tType = GEP->getSourceElementType();
        auto ret = ComplexGEPsInfoCache.insert(
            {GEPVarOffsetData, llvm::GetElementPtrInst::Create(tType, tPtr, PartialIdxs, "partial_gep")});
        DanglingGEP_it = ret.first;
    }
    llvm::Value *DanglingGEP = DanglingGEP_it->second;
    // Collect idxs. We will use them to find offset. Push zero constant first:
    // this will be needed for correct offset accumulation further.
    std::vector<llvm::Value *> Idxs = {llvm::ConstantInt::get(LLVMTypes::Int32Type, 0)};
    for (unsigned i = FirstConstIdx; i < GEP->getNumOperands(); ++i) {
        Idxs.push_back(GEP->getOperand(i));
    }
    // Get partial GEP type
    llvm::PointerType *PartialType = llvm::cast<llvm::PointerType>(DanglingGEP->getType());

    // Create temporary GEP that will help us to get some useful info
    llvm::Value *tPtr = llvm::ConstantPointerNull::get(PartialType);
    Assert(llvm::cast<llvm::GetElementPtrInst>(DanglingGEP));
    llvm::Type *tType = llvm::cast<llvm::GetElementPtrInst>(DanglingGEP)->getSourceElementType();

    llvm::GetElementPtrInst *GEPHelper = llvm::GetElementPtrInst::Create(tType, tPtr, Idxs);
    // Accumulate offset from helper
    llvm::APInt acc(g->target->is32Bit() ? 32 : 64, 0, true);
    bool checker = GEPHelper->accumulateConstantOffset(GEP->getModule()->getDataLayout(), acc);
    Assert(checker);
    // Finally, store data.
    res.Ptr = DanglingGEP;
    res.Offset = acc.getSExtValue();
    res.IsConstantOffset = true;

    return res;
}

// Analyse arithmetic calculations on pointer.
// TODO: not implemented, returns stopper
MemoryCoalescing::BasePtrInfo MemoryCoalescing::analyseArithmetics(llvm::BinaryOperator *Arithm) {
    BasePtrInfo res;
    res.IsConstantOffset = false;
    return res;
}

// Basic block optimization runner.
// TODO: runOnBasicBlock must call it to run optimization. See comment above.
void MemoryCoalescing::runOnBasicBlockImpl(llvm::BasicBlock &BB) {
    analyseInsts(BB);
    applyOptimization();
    deletePossiblyDeadInsts();
    clear();
}

void MemoryCoalescing::deletePossiblyDeadInsts() {
    for (auto *Inst : PossiblyDead) {
        if (Inst->use_empty())
            Inst->eraseFromParent();
    }
}

void MemoryCoalescing::clear() {
    BasePtrInfoCache.clear();
    ComplexGEPsInfoCache.clear();
    PossiblyDead.clear();
    Blocks.clear();
}

// Analyse instructions in BB. Gather them into optimizable blocks.
void MemoryCoalescing::analyseInsts(llvm::BasicBlock &BB) {
    auto bi = BB.begin(), be = BB.end();
    auto rbi = BB.rbegin(), rbe = BB.rend();
    Assert(OptType == MemType::OPT_LOAD || OptType == MemType::OPT_STORE);
    OptBlock CurrentBlock;
    for (; (OptType == MemType::OPT_LOAD) ? (bi != be) : (rbi != rbe);) {
        llvm::Instruction *Inst = NULL;
        if (OptType == MemType::OPT_LOAD) {
            Inst = &*bi;
            ++bi;
        } else {
            Inst = &*rbi;
            ++rbi;
        }

        if (isOptimizationTarget(Inst)) {
            // Find ptr and offsets
            BasePtrInfo BasePtr = findBasePtr(getPointer(Inst));
            OffsetsVecT Offsets = getOffset(Inst);
            if (BasePtr.IsConstantOffset && BasePtr.Ptr && !Offsets.empty()) {
                if (OptType == MemType::OPT_STORE && !CurrentBlock.PtrsData.empty()) {
                    Assert(CurrentBlock.PtrsData.size() == 1 && "Store coalescing can handle only one pointer at once");
                    if (CurrentBlock.PtrsData.find(BasePtr.Ptr) == CurrentBlock.PtrsData.end()) {
                        // Finish current block so the instruction is added to the new one
                        CurrentBlock = finishBlock(CurrentBlock);
                    }
                }

                // Recalculate offsets for BasePtr
                for (auto &Offset : Offsets)
                    Offset += BasePtr.Offset;
                // Add inst to block
                CurrentBlock.addInstruction(Inst, BasePtr.Ptr, Offsets,
                                            OptType == MemType::OPT_STORE ? getStoredValue(Inst) : nullptr);
                // Instruction that was added to the block won't block optimization
                continue;
            }
        }

        if (stopsOptimization(Inst)) {
            // Add current block and create new one
            CurrentBlock = finishBlock(CurrentBlock);
        }
    }

    // Add current block
    Blocks.push_back(CurrentBlock);
}

// Apply optimization for all optimizable blocks.
void MemoryCoalescing::applyOptimization() {
    for (auto &Block : Blocks) {
        for (auto &PD : Block.PtrsData) {
            // Apply dangling GEP
            if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(PD.first))
                if (!GEP->getParent())
                    GEP->insertBefore(PD.second.InsertPoint);
            // Run optimization
            optimizePtr(PD.first, PD.second, PD.second.InsertPoint);
        }
    }
}

// Check whether instruction blocks optimization.
// Optimization target checker should be called before because any store
// blocks optimization in general case.
bool MemoryCoalescing::stopsOptimization(llvm::Instruction *Inst) const {
    if (OptType == MemType::OPT_LOAD) {
        // For load coalescing, only stores can introduce problems.
        if (auto CI = llvm::dyn_cast<llvm::CallInst>(Inst)) {
            // Such call may introduce load-store-load sequence
            if (!CI->onlyReadsMemory())
                for (unsigned i = 0; i < Inst->getNumOperands(); ++i)
                    if (GetAddressSpace(Inst->getOperand(i)) == AddrSpace)
                        return true;
        } else if (auto SI = llvm::dyn_cast<llvm::StoreInst>(Inst)) {
            // May introduce load-store-load sequence
            if (GetAddressSpace(SI->getPointerOperand()) == AddrSpace)
                return true;
        }
    } else if (OptType == MemType::OPT_STORE) {
        // For store coalescing, both loads and stores can introduce problems.
        if (auto CI = llvm::dyn_cast<llvm::CallInst>(Inst)) {
            // Such call may introduce store-load/store-store sequence
            for (unsigned i = 0; i < Inst->getNumOperands(); ++i) {
                if (GetAddressSpace(Inst->getOperand(i)) == AddrSpace)
                    return true;
            }
        } else if (auto LI = llvm::dyn_cast<llvm::LoadInst>(Inst)) {
            // May introduce store-load-store sequence
            if (GetAddressSpace(LI->getPointerOperand()) == AddrSpace)
                return true;
        } else if (auto SI = llvm::dyn_cast<llvm::StoreInst>(Inst)) {
            // May introduce store-store-store sequence
            if (GetAddressSpace(SI->getPointerOperand()) == AddrSpace)
                return true;
        }
    } else
        Assert(0 && "Bad optimization type");

    return false;
}

// Store current block and init new one.
MemoryCoalescing::OptBlock MemoryCoalescing::finishBlock(MemoryCoalescing::OptBlock &Block) {
    Blocks.push_back(Block);
    return OptBlock();
}

// Get constant offset from vector.
MemoryCoalescing::OffsetsVecT MemoryCoalescing::getConstOffsetFromVector(llvm::Value *VecVal) const {
    Assert(VecVal && llvm::isa<llvm::VectorType>(VecVal->getType()) && "Expected vector type");
    OffsetsVecT res;
    auto ConstVec = llvm::dyn_cast<llvm::ConstantDataVector>(VecVal);

    if (!ConstVec)
        return res;
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    for (unsigned i = 0, size = llvm::cast<llvm::FixedVectorType>(ConstVec->getType())->getNumElements(); i < size;
         ++i) {
#else
    for (unsigned i = 0, size = llvm::cast<llvm::VectorType>(ConstVec->getType())->getNumElements(); i < size; ++i) {
#endif
        auto ConstantInt = llvm::dyn_cast<llvm::ConstantInt>(ConstVec->getElementAsConstant(i));
        if (!ConstantInt) {
            // Actually, we don't expect this to happen
            res.clear();
            break;
        }
        res.push_back(ConstantInt->getSExtValue());
    }

    return res;
}

// Apply scale on offsets.
void MemoryCoalescing::applyScale(MemoryCoalescing::OffsetsVecT &Offsets, MemoryCoalescing::OffsetT Scale) const {
    for (auto &Offset : Offsets)
        Offset *= Scale;
}

unsigned MemoryCoalescing::getScalarTypeSize(llvm::Type *Ty) const {
    Ty = Ty->getScalarType();
    if (Ty->isPointerTy())
        return g->target->is32Bit() ? 4 : 8;
    return Ty->getPrimitiveSizeInBits() >> 3;
}

llvm::Value *MemoryCoalescing::buildIEI(llvm::Value *InsertTo, llvm::Value *Val, MemoryCoalescing::OffsetT OffsetBytes,
                                        llvm::Instruction *InsertBefore) const {
    llvm::Type *ScalarType = InsertTo->getType()->getScalarType();
    llvm::Type *ValTy = Val->getType();

    Assert(!ValTy->isVectorTy() && !ValTy->isAggregateType() && "Expected scalar type");
    Assert(InsertTo->getType()->isVectorTy() && "Expected vector type");

    unsigned ScalarTypeBytes = getScalarTypeSize(ScalarType);
    unsigned ValTyBytes = getScalarTypeSize(ValTy);
    unsigned Idx = OffsetBytes / ScalarTypeBytes;
    unsigned Rem = OffsetBytes % ScalarTypeBytes;
    Idx = Rem ? Idx + 1 : Idx;
    llvm::Value *FinalInsertElement = Val;
    if (ValTy != ScalarType) {
        if (ValTyBytes == ScalarTypeBytes) {
            // Apply cast
            FinalInsertElement = buildCast(Val, ScalarType, InsertBefore);
        } else {
            // Need to create eei-cast-iei-cast chain.
            // Extract scalar type value
            auto *EEI = llvm::ExtractElementInst::Create(InsertTo, llvm::ConstantInt::get(LLVMTypes::Int64Type, Idx),
                                                         "mem_coal_diff_ty_eei", InsertBefore);
            // Cast it to vector of smaller types
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
            auto *Cast = buildCast(EEI, llvm::FixedVectorType::get(ValTy, ScalarTypeBytes / ValTyBytes), InsertBefore);
#else
            auto *Cast = buildCast(EEI, llvm::VectorType::get(ValTy, ScalarTypeBytes / ValTyBytes), InsertBefore);
#endif
            // Insert value into casted type. Do it via this builder so we don't duplicate logic of offset calculations.
            auto *IEI = buildIEI(Cast, Val, Rem, InsertBefore);
            // Cast to original type
            FinalInsertElement = buildCast(IEI, ScalarType, InsertBefore);
        }
    }

    return llvm::InsertElementInst::Create(
        InsertTo, FinalInsertElement, llvm::ConstantInt::get(LLVMTypes::Int64Type, Idx), "mem_coal_iei", InsertBefore);
}

llvm::Value *MemoryCoalescing::buildCast(llvm::Value *Val, llvm::Type *DstTy, llvm::Instruction *InsertBefore) const {
    // No cast needed: early return
    if (Val->getType() == DstTy)
        return Val;

    if (DstTy->isPointerTy() && !Val->getType()->isPointerTy()) {
        return new llvm::IntToPtrInst(Val, DstTy, "coal_diff_ty_ptr_cast", InsertBefore);
    } else if (!DstTy->isPointerTy() && Val->getType()->isPointerTy()) {
        auto *PtrToInt = new llvm::PtrToIntInst(Val, g->target->is32Bit() ? LLVMTypes::Int32Type : LLVMTypes::Int64Type,
                                                "coal_diff_ty_ptr_cast", InsertBefore);
        return buildCast(PtrToInt, DstTy, InsertBefore);
    } else {
        return new llvm::BitCastInst(Val, DstTy, "coal_diff_ty_cast", InsertBefore);
    }
}

llvm::Value *MemoryCoalescing::buildEEI(llvm::Value *ExtractFrom, MemoryCoalescing::OffsetT OffsetBytes,
                                        llvm::Type *DstTy, llvm::Instruction *InsertBefore) const {
    Assert(!DstTy->isVectorTy() && !DstTy->isAggregateType() && "Expected scalar type");
    Assert(ExtractFrom->getType()->isVectorTy() && "Expected vector type");

    llvm::Type *ScalarType = ExtractFrom->getType()->getScalarType();
    unsigned ScalarTypeBytes = getScalarTypeSize(ScalarType);
    unsigned DstTyBytes = getScalarTypeSize(DstTy);
    unsigned Idx = OffsetBytes / ScalarTypeBytes;
    unsigned Rem = OffsetBytes % ScalarTypeBytes;
    llvm::Value *Res = nullptr;

    if (Rem != 0) {
        // Unaligned case: the resulting value starts inside Idx element.
        // TODO: this is handled via shuffle vector. Actually, it can be done
        // for all cases, but it is possible that such shuffle vector would
        // introduce redundant instructions. This should be investigated
        // at least on Xe target.

        // Cast source to byte vector
        Res = buildCast(
            ExtractFrom,
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
            llvm::FixedVectorType::get(LLVMTypes::Int8Type,
                                       ScalarTypeBytes *
                                           llvm::cast<llvm::FixedVectorType>(ExtractFrom->getType())->getNumElements()),
#else
            llvm::VectorType::get(LLVMTypes::Int8Type,
                                  ScalarTypeBytes *
                                      llvm::cast<llvm::VectorType>(ExtractFrom->getType())->getNumElements()),
#endif
            InsertBefore);
        // Prepare Idxs vector for shuffle vector
        std::vector<unsigned int> ByteIdxs(DstTyBytes);
        OffsetT CurrIdx = OffsetBytes;
        for (auto &Val : ByteIdxs)
            Val = CurrIdx++;
        llvm::ArrayRef<unsigned int> ByteIdxsArg(ByteIdxs);
        // Extract bytes via shuffle vector
        Res = new llvm::ShuffleVectorInst(Res, llvm::UndefValue::get(Res->getType()),
                                          llvm::ConstantDataVector::get(*g->ctx, ByteIdxsArg), "coal_unaligned_loader",
                                          InsertBefore);
        // Cast byte vector to scalar value
        Res = buildCast(Res, llvm::IntegerType::get(*g->ctx, /* NumBits */ DstTyBytes << 3), InsertBefore);
        // Cast to actual DstTy
        return buildCast(Res, DstTy, InsertBefore);
    }

    Res = llvm::ExtractElementInst::Create(ExtractFrom, llvm::ConstantInt::get(LLVMTypes::Int64Type, Idx),
                                           "mem_coal_eei", InsertBefore);
    if (DstTy == ScalarType) {
        // Done here
        return Res;
    } else if (DstTyBytes == ScalarTypeBytes) {
        // Just create bitcast
        return buildCast(Res, DstTy, InsertBefore);
    }

    // Smaller type case. Need to insert cast-eei chain.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    auto Cast = buildCast(Res, llvm::FixedVectorType::get(DstTy, ScalarTypeBytes / DstTyBytes), InsertBefore);
#else
    auto Cast = buildCast(Res, llvm::VectorType::get(DstTy, ScalarTypeBytes / DstTyBytes), InsertBefore);
#endif
    // Now call the builder with adjusted types
    return buildEEI(Cast, Rem, DstTy, InsertBefore);
}

llvm::Value *MemoryCoalescing::extractValueFromBlock(const MemoryCoalescing::BlockInstsVecT &BlockInstsVec,
                                                     MemoryCoalescing::OffsetT OffsetBytes, llvm::Type *DstTy,
                                                     llvm::Instruction *InsertBefore) const {
    Assert(BlockInstsVec.size() > 0);
    OffsetT BlockSizeInBytes = getScalarTypeSize(BlockInstsVec[0]->getType()) *
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
                               llvm::cast<llvm::FixedVectorType>(BlockInstsVec[0]->getType())->getNumElements();
#else
                               llvm::cast<llvm::VectorType>(BlockInstsVec[0]->getType())->getNumElements();
#endif
    unsigned StartIdx = OffsetBytes / BlockSizeInBytes;
    unsigned EndIdx = (OffsetBytes + getScalarTypeSize(DstTy) - 1) / BlockSizeInBytes;
    unsigned BlocksAffected = EndIdx - StartIdx + 1;
    Assert(EndIdx < BlockInstsVec.size());
    if (BlocksAffected == 1) {
        // Simple case: just get value needed
        return buildEEI(BlockInstsVec[StartIdx], OffsetBytes % BlockSizeInBytes, DstTy, InsertBefore);
    } else {
        // Need to get value from several blocks
        llvm::Value *ByteVec = llvm::UndefValue::get(
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
            llvm::FixedVectorType::get(LLVMTypes::Int8Type, getScalarTypeSize(DstTy)));
#else
            llvm::VectorType::get(LLVMTypes::Int8Type, getScalarTypeSize(DstTy)));
#endif
        for (OffsetT CurrOffset = OffsetBytes, TargetOffset = 0; CurrOffset < OffsetBytes + getScalarTypeSize(DstTy);
             ++CurrOffset, ++TargetOffset) {
            unsigned Idx = CurrOffset / BlockSizeInBytes;
            unsigned LocalOffset = CurrOffset % BlockSizeInBytes;
            llvm::Value *Elem = buildEEI(BlockInstsVec[Idx], LocalOffset, LLVMTypes::Int8Type, InsertBefore);
            ByteVec = buildIEI(ByteVec, Elem, TargetOffset, InsertBefore);
        }
        llvm::Value *ScalarizedByteVec = buildCast(
            ByteVec, llvm::IntegerType::get(*g->ctx, /* NumBits */ getScalarTypeSize(DstTy) << 3), InsertBefore);
        return buildCast(ScalarizedByteVec, DstTy, InsertBefore);
    }
}

class XeGatherCoalescing : public MemoryCoalescing {
  private:
    bool isOptimizationTarget(llvm::Instruction *Inst) const;
    llvm::Value *getPointer(llvm::Instruction *Inst) const;
    OffsetsVecT getOffset(llvm::Instruction *Inst) const;
    llvm::Value *getStoredValue(llvm::Instruction *Inst) const { return nullptr; }
    void optimizePtr(llvm::Value *Ptr, PtrData &PD, llvm::Instruction *InsertPoint);
    void runOnBasicBlock(llvm::BasicBlock &BB);

    llvm::CallInst *getPseudoGatherConstOffset(llvm::Instruction *Inst) const;
    bool isConstOffsetPseudoGather(llvm::CallInst *CI) const;

  public:
    static char ID;
    XeGatherCoalescing() : MemoryCoalescing(ID, MemoryCoalescing::MemType::OPT_LOAD, AddressSpace::ispc_global) {}

    llvm::StringRef getPassName() const { return "Xe Gather Coalescing"; }
};

char XeGatherCoalescing::ID = 0;

void XeGatherCoalescing::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("XeGatherCoalescing");
    runOnBasicBlockImpl(bb);
    DEBUG_END_PASS("XeGatherCoalescing");
}

void XeGatherCoalescing::optimizePtr(llvm::Value *Ptr, PtrData &PD, llvm::Instruction *InsertPoint) {
    // Analyse memory accesses
    OffsetT MinIdx = INT64_MAX;
    OffsetT MaxIdx = INT64_MIN;
    unsigned TotalMemOpsCounter = PD.InstsData.size();
    llvm::Type *LargestType = nullptr;
    unsigned LargestTypeSize = 0;
    for (auto &ID : PD.InstsData) {
        // Adjust borders
        for (auto Idx : ID.Offsets) {
            MaxIdx = std::max(Idx, MaxIdx);
            MinIdx = std::min(Idx, MinIdx);
        }
        // Largest type is needed to handle the case with different type sizes
        unsigned TypeSize = getScalarTypeSize(ID.Inst->getType());
        if (TypeSize > LargestTypeSize) {
            LargestTypeSize = TypeSize;
            LargestType = ID.Inst->getType()->getScalarType();
            if (LargestType->isPointerTy())
                LargestType = g->target->is32Bit() ? LLVMTypes::Int32Type : LLVMTypes::Int64Type;
        }
    }

    // Calculate data length
    Assert(LargestTypeSize > 0);
    uint64_t DataSize = MaxIdx - MinIdx + LargestTypeSize;
    // Calculate size of block loads in powers of two:
    // block loads are aligned to OWORDs
    unsigned ReqSize = 1;
    while (ReqSize < DataSize)
        ReqSize <<= 1;

    // Adjust req size and calculate num if insts needed
    // TODO: experiment showed performance improvement with
    // max ReqSize of 4 * OWORD instead of 8 * OWORD.
    // Further investigation is needed.
    unsigned MemInstsNeeded = 1;
    if (ReqSize > 4 * OWORD) {
        // Dealing with powers of two
        MemInstsNeeded = ReqSize / (4 * OWORD);
        ReqSize = 4 * OWORD;
    }

    // TODO: not clear if we should skip it
    if (ReqSize < OWORD) {
        // Skip for now.
        return;
    }

    // Check for threshold
    int MemOpsDiff = TotalMemOpsCounter - MemInstsNeeded;
    if (MemOpsDiff < g->opt.thresholdForXeGatherCoalescing) {
        return;
    }

    // Build block loads
    BlockInstsVecT BlockLDs;
    for (unsigned i = 0; i < MemInstsNeeded; ++i) {
        llvm::Constant *Offset = llvm::ConstantInt::get(LLVMTypes::Int64Type, MinIdx + i * ReqSize);
        llvm::PtrToIntInst *PtrToInt =
            new llvm::PtrToIntInst(Ptr, LLVMTypes::Int64Type, "vectorized_ptrtoint", InsertPoint);
        llvm::Instruction *Addr = llvm::BinaryOperator::CreateAdd(PtrToInt, Offset, "vectorized_address", InsertPoint);
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
        llvm::Type *RetType = llvm::FixedVectorType::get(LargestType, ReqSize / LargestTypeSize);
#else
        llvm::Type *RetType = llvm::VectorType::get(LargestType, ReqSize / LargestTypeSize);
#endif
        llvm::Instruction *LD = nullptr;
        if (g->opt.buildLLVMLoadsOnXeGatherCoalescing) {
            // Experiment: build standard llvm load instead of block ld
            llvm::IntToPtrInst *PtrForLd =
                new llvm::IntToPtrInst(Addr, llvm::PointerType::get(RetType, 0), "vectorized_address_ptr", InsertPoint);
            LD = new llvm::LoadInst(RetType, PtrForLd, "vectorized_ld_exp", InsertPoint);
        } else {
            llvm::Function *Fn = llvm::GenXIntrinsic::getGenXDeclaration(
                m->module, llvm::GenXIntrinsic::genx_svm_block_ld_unaligned, {RetType, Addr->getType()});
            LD = llvm::CallInst::Create(Fn, {Addr}, "vectorized_ld", InsertPoint);
        }
        BlockLDs.push_back(LD);
    }

    // Replace users
    for (auto &ID : PD.InstsData) {
        if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(ID.Inst)) {
            // Adjust offset
            OffsetT Offset = ID.Offsets[0] - MinIdx;
            LI->replaceAllUsesWith(extractValueFromBlock(BlockLDs, Offset, LI->getType(), InsertPoint));
        } else if (auto *Gather = getPseudoGatherConstOffset(ID.Inst)) {
            llvm::Value *NewVal = llvm::UndefValue::get(Gather->getType());
            unsigned CurrElem = 0;
            for (auto Offset : ID.Offsets) {
                // Adjust offset
                OffsetT AdjOffset = Offset - MinIdx;
                auto *ExtractedValue =
                    extractValueFromBlock(BlockLDs, AdjOffset, Gather->getType()->getScalarType(), InsertPoint);
                NewVal = llvm::InsertElementInst::Create(NewVal, ExtractedValue,
                                                         llvm::ConstantInt::get(LLVMTypes::Int64Type, CurrElem),
                                                         "gather_coal_iei", InsertPoint);
                ++CurrElem;
            }
            Gather->replaceAllUsesWith(NewVal);
        }

        // Mark to delete
        PossiblyDead.insert(ID.Inst);
    }

    // Done
    modifiedAny = true;
}

llvm::CallInst *XeGatherCoalescing::getPseudoGatherConstOffset(llvm::Instruction *Inst) const {
    if (auto CI = llvm::dyn_cast<llvm::CallInst>(Inst)) {
        llvm::Function *Function = CI->getCalledFunction();
        if (Function && Function->getName().startswith("__pseudo_gather_base_offsets"))
            if (isConstOffsetPseudoGather(CI))
                return CI;
    }
    return nullptr;
}

bool XeGatherCoalescing::isConstOffsetPseudoGather(llvm::CallInst *CI) const {
    Assert(CI != nullptr && CI->getCalledFunction() &&
           CI->getCalledFunction()->getName().startswith("__pseudo_gather_base_offsets"));
    llvm::Value *opOffset = CI->getOperand(2);

    return (llvm::isa<llvm::ConstantDataVector>(opOffset) || llvm::isa<llvm::ConstantAggregateZero>(opOffset) ||
            llvm::isa<llvm::ConstantVector>(opOffset));
}

bool XeGatherCoalescing::isOptimizationTarget(llvm::Instruction *Inst) const {
    if (auto LI = llvm::dyn_cast<llvm::LoadInst>(Inst)) {
        if (!LI->getType()->isVectorTy() && !LI->getType()->isAggregateType())
            return GetAddressSpace(LI->getPointerOperand()) == AddrSpace;
    } else if (auto Gather = getPseudoGatherConstOffset(Inst)) {
        return GetAddressSpace(getPointer(Gather)) == AddrSpace;
    }

    return false;
}

MemoryCoalescing::OffsetsVecT XeGatherCoalescing::getOffset(llvm::Instruction *Inst) const {
    OffsetsVecT Res;

    if (llvm::isa<llvm::LoadInst>(Inst))
        Res.push_back(0);
    else if (auto Gather = getPseudoGatherConstOffset(Inst)) {
        if (llvm::isa<llvm::ConstantAggregateZero>(Gather->getOperand(2))) {
            Res.push_back(0);
        } else {
            Res = getConstOffsetFromVector(Gather->getOperand(2));
            applyScale(Res, llvm::cast<llvm::ConstantInt>(Gather->getOperand(1))->getSExtValue());
        }
    }

    return Res;
}

llvm::Value *XeGatherCoalescing::getPointer(llvm::Instruction *Inst) const {
    if (auto LI = llvm::dyn_cast<llvm::LoadInst>(Inst)) {
        return LI->getPointerOperand();
    } else if (auto Gather = getPseudoGatherConstOffset(Inst)) {
        return Gather->getOperand(0);
    }

    return nullptr;
}

static llvm::Pass *CreateXeGatherCoalescingPass() { return new XeGatherCoalescing; }

///////////////////////////////////////////////////////////////////////////
// ReplaceLLVMIntrinsics

/** This pass replaces LLVM intrinsics unsupported on Xe
 */

class ReplaceLLVMIntrinsics : public llvm::FunctionPass {
  public:
    static char ID;
    ReplaceLLVMIntrinsics() : FunctionPass(ID) {}
    llvm::StringRef getPassName() const { return "LLVM intrinsics replacement"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);
};

char ReplaceLLVMIntrinsics::ID = 0;

bool ReplaceLLVMIntrinsics::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("LLVM intrinsics replacement");
    std::vector<llvm::AllocaInst *> Allocas;

    bool modifiedAny = false;

restart:
    for (llvm::BasicBlock::iterator I = bb.begin(), E = --bb.end(); I != E; ++I) {
        llvm::Instruction *inst = &*I;
        if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst)) {
            llvm::Function *func = ci->getCalledFunction();
            if (func == NULL || !func->isIntrinsic())
                continue;

            if (func->getName().equals("llvm.trap")) {
                llvm::Type *argTypes[] = {LLVMTypes::Int1VectorType, LLVMTypes::Int16VectorType};
                // Description of parameters for genx_raw_send_noresult can be found in target-genx.ll
                auto Fn = llvm::GenXIntrinsic::getGenXDeclaration(
                    m->module, llvm::GenXIntrinsic::genx_raw_send_noresult, argTypes);
                llvm::SmallVector<llvm::Value *, 8> Args;
                Args.push_back(llvm::ConstantInt::get(LLVMTypes::Int32Type, 0));
                Args.push_back(llvm::ConstantVector::getSplat(
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
                    g->target->getNativeVectorWidth(),
#elif ISPC_LLVM_VERSION < ISPC_LLVM_12_0
                    {static_cast<unsigned int>(g->target->getNativeVectorWidth()), false},
#else // LLVM 12.0+
                    llvm::ElementCount::get(static_cast<unsigned int>(g->target->getNativeVectorWidth()), false),
#endif
                    llvm::ConstantInt::getTrue(*g->ctx)));

                Args.push_back(llvm::ConstantInt::get(LLVMTypes::Int32Type, 39));
                Args.push_back(llvm::ConstantInt::get(LLVMTypes::Int32Type, 33554448));
                llvm::Value *zeroMask = llvm::ConstantVector::getSplat(
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
                    g->target->getNativeVectorWidth(),
#elif ISPC_LLVM_VERSION < ISPC_LLVM_12_0
                    {static_cast<unsigned int>(g->target->getNativeVectorWidth()), false},
#else // LLVM 12.0+
                    llvm::ElementCount::get(static_cast<unsigned int>(g->target->getNativeVectorWidth()), false),
#endif
                    llvm::Constant::getNullValue(llvm::Type::getInt16Ty(*g->ctx)));
                Args.push_back(zeroMask);

                llvm::Instruction *newInst = llvm::CallInst::Create(Fn, Args, ci->getName());
                if (newInst != NULL) {
                    llvm::ReplaceInstWithInst(ci, newInst);
                    modifiedAny = true;
                    goto restart;
                }
            } else if (func->getName().equals("llvm.experimental.noalias.scope.decl")) {
                // These intrinsics are not supported by backend so remove them.
                ci->eraseFromParent();
                modifiedAny = true;
                goto restart;
            } else if (func->getName().contains("llvm.abs")) {
                // Replace llvm.asb with llvm.genx.aba.alternative
                Assert(ci->getOperand(0));
                llvm::Type *argType = ci->getOperand(0)->getType();

                llvm::Type *Tys[2];
                Tys[0] = func->getReturnType(); // return type
                Tys[1] = argType;               // value type

                llvm::GenXIntrinsic::ID xeAbsID =
                    argType->isIntOrIntVectorTy() ? llvm::GenXIntrinsic::genx_absi : llvm::GenXIntrinsic::genx_absf;
                auto Fn = llvm::GenXIntrinsic::getGenXDeclaration(m->module, xeAbsID, Tys);
                Assert(Fn);
                llvm::Instruction *newInst = llvm::CallInst::Create(Fn, ci->getOperand(0), "");
                if (newInst != NULL) {
                    LLVMCopyMetadata(newInst, ci);
                    llvm::ReplaceInstWithInst(ci, newInst);
                    modifiedAny = true;
                    goto restart;
                }
            }
        }
    }
    DEBUG_END_PASS("LLVM intrinsics replacement");
    return modifiedAny;
}

bool ReplaceLLVMIntrinsics::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("ReplaceLLVMIntrinsics::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateReplaceLLVMIntrinsics() { return new ReplaceLLVMIntrinsics(); }

///////////////////////////////////////////////////////////////////////////
// CheckIRForGenTarget

/** This pass checks IR for Xe target and fix arguments for Xe intrinsics if needed.
    In case if unsupported statement is found, it reports error and stops compilation.
    Currently it performs 2 checks:
    1. double type support by target
    2. prefetch support by target and fixing prefetch args
 */

class CheckIRForGenTarget : public llvm::FunctionPass {
  public:
    static char ID;
    CheckIRForGenTarget(bool last = false) : FunctionPass(ID) {}

    llvm::StringRef getPassName() const { return "Check IR for Xe target"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);
};

char CheckIRForGenTarget::ID = 0;

bool CheckIRForGenTarget::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("CheckIRForGenTarget");
    bool modifiedAny = false;
    // This list contains regex expr for unsupported function names
    // To be extended

    for (llvm::BasicBlock::iterator I = bb.begin(), E = --bb.end(); I != E; ++I) {
        llvm::Instruction *inst = &*I;
        SourcePos pos;
        LLVMGetSourcePosFromMetadata(inst, &pos);
        if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst)) {
            if (llvm::GenXIntrinsic::getGenXIntrinsicID(ci) == llvm::GenXIntrinsic::genx_lsc_prefetch_stateless) {
                // If prefetch is supported, fix data size parameter
                Assert(ci->arg_size() > 6);
                llvm::Value *dataSizeVal = ci->getArgOperand(6);
                llvm::ConstantInt *dataSizeConst = llvm::dyn_cast<llvm::ConstantInt>(dataSizeVal);
                Assert(dataSizeConst && (dataSizeConst->getBitWidth() == 8));
                int dataSizeNum = dataSizeConst->getSExtValue();
                // 0: invalid
                // 1: d8
                // 2: d16
                // 3: d32
                // 4: d64
                // Valid user's input is 1, 2, 4, 8
                int8_t genSize = 3;
                switch (dataSizeNum) {
                case 1:
                    genSize = 1;
                    break;
                case 2:
                    genSize = 2;
                    break;
                case 4:
                    genSize = 3;
                    break;
                case 8:
                    genSize = 4;
                    break;
                default:
                    Error(pos, "Incorrect data size argument for \'prefetch\'. Valid values are 1, 2, 4, 8");
                }
                llvm::Value *dataSizeGen = llvm::ConstantInt::get(LLVMTypes::Int8Type, genSize);
                ci->setArgOperand(6, dataSizeGen);
            }
        }
        // Report error if double type is not supported by the target
        if (!g->target->hasFp64Support()) {
            for (int i = 0; i < (int)inst->getNumOperands(); ++i) {
                llvm::Type *t = inst->getOperand(i)->getType();
                if (t == LLVMTypes::DoubleType || t == LLVMTypes::DoublePointerType ||
                    t == LLVMTypes::DoubleVectorType || t == LLVMTypes::DoubleVectorPointerType) {
                    Error(pos, "\'double\' type is not supported by the target\n");
                }
            }
        }
    }
    DEBUG_END_PASS("CheckIRForGenTarget");

    return modifiedAny;
}

bool CheckIRForGenTarget::runOnFunction(llvm::Function &F) {
    llvm::TimeTraceScope FuncScope("CheckIRForGenTarget::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateCheckIRForGenTarget() { return new CheckIRForGenTarget(); }

///////////////////////////////////////////////////////////////////////////
// MangleOpenCLBuiltins

/** This pass mangles SPIR-V OpenCL builtins used in Xe target file
 */

class MangleOpenCLBuiltins : public llvm::FunctionPass {
  public:
    static char ID;
    MangleOpenCLBuiltins(bool last = false) : FunctionPass(ID) {}

    llvm::StringRef getPassName() const { return "Mangle OpenCL builtins"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);
};

char MangleOpenCLBuiltins::ID = 0;

static std::string mangleMathOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName().startswith("__spirv_ocl") && "wrong argument: ocl builtin is expected");
    std::string mangledName;
    llvm::Type *retType = func.getReturnType();
    std::string funcName = func.getName().str();
    std::vector<llvm::Type *> ArgTy;
    // spirv OpenCL builtins are used for double types only
    Assert(retType->isVectorTy() && llvm::dyn_cast<llvm::FixedVectorType>(retType)->getElementType()->isDoubleTy() ||
           retType->isSingleValueType() && retType->isDoubleTy());
    if (retType->isVectorTy() && llvm::dyn_cast<llvm::FixedVectorType>(retType)->getElementType()->isDoubleTy()) {
        // Get vector width from retType. Required width may be different from target width
        // for example for 32-width targets
        ArgTy.push_back(llvm::FixedVectorType::get(LLVMTypes::DoubleType,
                                                   llvm::dyn_cast<llvm::FixedVectorType>(retType)->getNumElements()));
        // _DvWIDTH suffix is used in target file to differentiate scalar
        // and vector versions of intrinsics. Here we remove this
        // suffix and mangle the name.
        size_t pos = funcName.find("_DvWIDTH");
        if (pos != std::string::npos) {
            funcName.erase(pos, 8);
        }
    } else if (retType->isSingleValueType() && retType->isDoubleTy()) {
        ArgTy.push_back(LLVMTypes::DoubleType);
    }
    mangleOpenClBuiltin(funcName, ArgTy, mangledName);
    return mangledName;
}

static std::string manglePrintfOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName() == "__spirv_ocl_printf" && "wrong argument: ocl builtin is expected");
    std::string mangledName;
    mangleOpenClBuiltin(func.getName().str(), func.getArg(0)->getType(), mangledName);
    return mangledName;
}

static std::string mangleOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName().startswith("__spirv_ocl") && "wrong argument: ocl builtin is expected");
    if (func.getName() == "__spirv_ocl_printf")
        return manglePrintfOCLBuiltin(func);
    return mangleMathOCLBuiltin(func);
}

bool MangleOpenCLBuiltins::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("MangleOpenCLBuiltins");
    bool modifiedAny = false;
    for (llvm::BasicBlock::iterator I = bb.begin(), E = --bb.end(); I != E; ++I) {
        llvm::Instruction *inst = &*I;
        if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst)) {
            llvm::Function *func = ci->getCalledFunction();
            if (func == NULL)
                continue;
            if (func->getName().startswith("__spirv_ocl")) {
                std::string mangledName = mangleOCLBuiltin(*func);
                func->setName(mangledName);
                modifiedAny = true;
            }
        }
    }
    DEBUG_END_PASS("MangleOpenCLBuiltins");

    return modifiedAny;
}

bool MangleOpenCLBuiltins::runOnFunction(llvm::Function &F) {
    llvm::TimeTraceScope FuncScope("MangleOpenCLBuiltins::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateMangleOpenCLBuiltins() { return new MangleOpenCLBuiltins(); }

class DemotePHIs : public llvm::FunctionPass {
  public:
    static char ID;
    DemotePHIs() : FunctionPass(ID) {}
    llvm::StringRef getPassName() const { return "Demote PHI nodes"; }
    bool runOnFunction(llvm::Function &F);
};

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

static llvm::Pass *CreateDemotePHIs() { return new DemotePHIs(); }

#endif
