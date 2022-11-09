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
    @brief Implementation of ISPC optimization pipeline.
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

#ifdef ISPC_XE_ENABLED
static llvm::Pass *CreateDemotePHIs();
static llvm::Pass *CreateCheckIRForGenTarget();
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

#ifdef ISPC_XE_ENABLED
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
